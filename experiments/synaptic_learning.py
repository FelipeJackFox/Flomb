"""Differentiable activity where the connectome's own gains learn and the interface stays fixed."""
import torch
from torch.nn import functional as F
from experiments.plastic_sparse import PlasticMessage

class SynapticBrain(torch.nn.Module):
    """Same forward as spatial_decoder.activity_map, with gradients into edge and node gains only."""
    def __init__(self,brain,head):
        super().__init__();self.brain,self.head=brain,head
        brain.requires_grad_(False);head.requires_grad_(False)
        brain.edge_log_gain.requires_grad_(True);brain.node_log_gain.requires_grad_(True)
    def synapses(self):return [self.brain.edge_log_gain,self.brain.node_log_gain]
    def activity(self,x,context):
        b=self.brain;count=len(x);sizes=(context[:,0]*16).round().long();mapped=sorted(set(sizes.tolist()))
        for size in mapped:
            if not 2<=size<=16 or not hasattr(b,f'out_{size}'):raise ValueError(f'Missing output mapping for {size}x{size}')
        grid=x.reshape(count,16,16,10).permute(0,3,1,2)
        with torch.no_grad():
            encoded=b.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],1))*(grid.sum(1,keepdim=True)!=0)
            sg=2*b.input_xy[None]*(sizes[:,None,None]-1)/15-1
            samples=F.grid_sample(encoded,sg[:,:,None],align_corners=True).squeeze(-1)
            sensory=samples[:,b.input_channel,torch.arange(len(b.input_channel))]
            drive=x.new_zeros(b.operator.shape[0],count).index_copy(0,b.input_index,sensory.T)
        state=torch.tanh(drive);values=b.base*b.edge_log_gain.clamp(-2,2).exp();gain=b.node_log_gain.clamp(-1,1).exp()[:,None]
        for _ in range(b.cycles):state=torch.tanh(PlasticMessage.apply(values,gain*state,b.operator)+.15*drive)
        result=x.new_zeros(count,10,16,16)
        for size in mapped:
            selected=torch.nonzero(sizes==size).flatten();ix,wt=getattr(b,f'out_{size}'),getattr(b,f'weight_{size}')
            pooled=(state[ix][:,:,:,selected]*wt[:,:,:,None]).sum(2).permute(2,1,0)
            result[selected,:,:size,:size]=pooled.reshape(len(selected),10,size,size)*10
        return result
    def forward(self,x,context):return self.head(self.activity(x,context),context)
