"""CPU diagnostic models. SciPy retains the complete graph; torch learns transforms.

This is an architecture probe, not a replacement checkpoint format for train.py.
Only first-order derivatives are supported by the sparse bridge.
"""
import numpy as np
import torch
from torch import nn


class SparseMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, operator):
        if state.device.type != 'cpu':
            raise ValueError('SciPy bridge requires CPU tensors')
        ctx.operator = operator
        return torch.from_numpy(operator.multiply(state.detach().contiguous().numpy()))

    @staticmethod
    def backward(ctx, gradient):
        result = ctx.operator.multiply(gradient.detach().contiguous().numpy(), True)
        return torch.from_numpy(result), None


class ConnectomePolicy(nn.Module):
    def __init__(self, operator, expressive=False, channels=2):
        super().__init__()
        self.operator = operator
        self.expressive = expressive
        self.channels = channels if expressive else 1
        n = len(operator.log_gain)
        self.register_buffer('indices', torch.from_numpy(np.concatenate(
            [operator.input_index, operator.extra_input_index], axis=1)).long())
        self.register_buffer('signs', torch.from_numpy(np.concatenate(
            [operator.input_sign, operator.extra_input_sign], axis=1)))
        self.register_buffer('outputs', torch.from_numpy(operator.output_index).long())
        self.log_gain = nn.Parameter(torch.zeros(n, self.channels))
        if expressive:
            self.encoder = nn.Sequential(nn.Conv2d(10, 16, 3, padding=1), nn.Tanh(),
                                         nn.Conv2d(16, 10*self.channels, 3, padding=1))
            self.bias = nn.Parameter(torch.zeros(n, self.channels))
            self.mix = nn.Parameter(torch.eye(self.channels))
            self.retention = nn.Parameter(torch.full((self.channels,), -2.0))
        self.head = nn.Linear(len(operator.output_index)*self.channels+2, 256)
        nn.init.normal_(self.head.weight, std=.01 / np.sqrt(self.head.in_features))
        nn.init.zeros_(self.head.bias)

    def features(self, x):
        batch = len(x)
        if self.expressive:
            grid = x.reshape(batch, 16, 16, 10).permute(0, 3, 1, 2)
            valid = (grid.sum(1, keepdim=True) != 0)
            encoded = self.encoder(grid).reshape(batch, self.channels, 10, 16, 16)
            encoded = (encoded + grid[:, None]) * valid[:, None]
            encoded = encoded.permute(0, 3, 4, 2, 1).reshape(batch, 2560, self.channels)
        else:
            encoded = x[..., None]
        # Chunk by projection slot to avoid a [batch, neurons, slots, channels] tensor.
        drive = encoded.new_zeros(len(self.log_gain), batch, self.channels)
        for slot in range(self.indices.shape[1]):
            drive = drive + encoded[:, self.indices[:, slot]].permute(1, 0, 2) * self.signs[:, slot, None, None]
        h = torch.tanh(drive)
        gain = self.log_gain.clamp(-1, 1).exp()[:, None]
        for _ in range(self.operator.cycles):
            message = SparseMessage.apply((gain*h).reshape(len(h), -1), self.operator).reshape_as(h)
            if self.expressive:
                candidate = torch.tanh(message @ self.mix + self.bias[:, None] + .1*drive)
                retention = self.retention.sigmoid()
                h = retention*h + (1-retention)*candidate
            else:
                h = torch.tanh(message)
        return h[self.outputs].permute(1, 0, 2).reshape(batch, -1)*self.operator.feature_scale

    def forward(self, x, context):
        return self.head(torch.cat([self.features(x), context], dim=1))


class ConvPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv2d(32, 1, 1))

    def forward(self, x, context):
        grid = x.reshape(-1, 16, 16, 10).permute(0, 3, 1, 2)
        return self.net(torch.cat([grid, context[:, :, None, None].expand(-1, -1, 16, 16)], 1)).flatten(1)


def equivalent_loss(logits, legal, labels):
    """Uniform teacher distribution over every certified safe action, as train.py."""
    if not torch.all(labels.any(1)) or torch.any(labels & ~legal):
        raise ValueError('Every row needs a nonempty legal teacher action set')
    target = labels.to(logits.dtype)
    target = target/target.sum(1, keepdim=True)
    return -(target * logits.masked_fill(~legal, -1e9).log_softmax(1)).sum(1).mean()
