"""Differentiable spatial activity with immutable connectome coefficients.

Only encoder and spatial head learn. Fixed CSR blocks are bound once, avoiding
edge-gradient allocation and repeated construction of the transposed graph.
"""
import torch
from torch.nn import functional as F

class FixedMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, operator, values):
        ctx.operator, ctx.values = operator, values
        return torch.from_numpy(operator.multiply(values, x.detach().contiguous().numpy()))

    @staticmethod
    def backward(ctx, dy):
        return torch.from_numpy(ctx.operator.multiply(ctx.values, dy.detach().contiguous().numpy(), True)), None, None

class JointInterface(torch.nn.Module):
    def __init__(self, brain, head):
        super().__init__()
        self.brain, self.head = brain, head
        brain.requires_grad_(False)
        brain.encoder.requires_grad_(True)
        self.values = (brain.base*(brain.edge_log_gain.clamp(-2,2).exp() if brain.plastic else 1)).detach().numpy().copy()
        self.gain = brain.node_log_gain.detach().clamp(-1,1).exp()[:,None]
        brain.operator.bind(self.values, force=True)

    def activity(self, x, context, dense=None):
        b = self.brain; count = len(x); sizes = (context[:,0]*16).round().long()
        mapped_sizes = sorted(set(sizes.tolist()))
        for size in mapped_sizes:
            if not 2 <= size <= 16 or not hasattr(b,f'out_{size}') or not hasattr(b,f'weight_{size}'):
                raise ValueError(f'Missing output mapping for {size}x{size}')
        grid = x.reshape(count,16,16,10).permute(0,3,1,2)
        encoded = b.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],1))
        encoded = encoded*(grid.sum(1,keepdim=True)!=0)
        sg = 2*b.input_xy[None]*(sizes[:,None,None]-1)/15-1
        samples = F.grid_sample(encoded,sg[:,:,None],align_corners=True).squeeze(-1)
        sensory = samples[:,b.input_channel,torch.arange(len(b.input_channel))]
        drive = x.new_zeros(b.operator.shape[0],count).index_copy(0,b.input_index,sensory.T)
        state = torch.tanh(drive)
        for _ in range(b.cycles):
            signal = FixedMessage.apply(self.gain*state,b.operator,self.values) if dense is None else dense@(self.gain*state)
            state = torch.tanh(signal+.15*drive)
        result = x.new_zeros(count,10,16,16)
        for size in mapped_sizes:
            selected = torch.nonzero(sizes==size).flatten()
            if not len(selected): continue
            ix, wt = getattr(b,f'out_{size}'), getattr(b,f'weight_{size}')
            pooled = (state[ix][:,:,:,selected]*wt[:,:,:,None]).sum(2).permute(2,1,0)
            result[selected,:,:size,:size] = pooled.reshape(len(selected),10,size,size)*10
        return result

    def forward(self,x,context):
        return self.head(self.activity(x,context),context)
