"""One-cycle activity with a trainable entrance: encoder always, first-hop synapses optionally."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.plastic_sparse import PlasticMessage
from experiments.joint_interface import FixedMessage

class OpenBrain(nn.Module):
    def __init__(self,brain,synapses):
        super().__init__();self.brain,self.plastic=brain,synapses
        brain.requires_grad_(False);brain.encoder.requires_grad_(True)
        if synapses:brain.edge_log_gain.requires_grad_(True);brain.node_log_gain.requires_grad_(True)
        else:
            self.values=(brain.base*brain.edge_log_gain.clamp(-2,2).exp()).detach().numpy().copy();brain.operator.bind(self.values,force=True)
    def synapses(self):return [self.brain.edge_log_gain,self.brain.node_log_gain] if self.plastic else []
    def activity(self,x,context):
        b=self.brain;count=len(x);sizes=(context[:,0]*16).round().long();mapped=sorted(set(sizes.tolist()))
        grid=x.reshape(count,16,16,10).permute(0,3,1,2)
        encoded=b.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],1))*(grid.sum(1,keepdim=True)!=0)
        sg=2*b.input_xy[None]*(sizes[:,None,None]-1)/15-1
        samples=F.grid_sample(encoded,sg[:,:,None],align_corners=True).squeeze(-1)
        sensory=samples[:,b.input_channel,torch.arange(len(b.input_channel))]
        drive=x.new_zeros(b.operator.shape[0],count).index_copy(0,b.input_index,sensory.T)
        state=torch.tanh(drive);gain=b.node_log_gain.clamp(-1,1).exp()[:,None]
        values=b.base*b.edge_log_gain.clamp(-2,2).exp() if self.plastic else None
        for _ in range(b.cycles):
            signal=PlasticMessage.apply(values,gain*state,b.operator) if self.plastic else FixedMessage.apply(gain*state,b.operator,self.values)
            state=torch.tanh(signal+.15*drive)
        result=x.new_zeros(count,10,16,16)
        for size in mapped:
            selected=torch.nonzero(sizes==size).flatten();ix,wt=getattr(b,f'out_{size}'),getattr(b,f'weight_{size}')
            pooled=(state[ix][:,:,:,selected]*wt[:,:,:,None]).sum(2).permute(2,1,0)
            result[selected,:,:size,:size]=pooled.reshape(len(selected),10,size,size)*10
        return result

class ClueDecoder(nn.Module):
    """Auxiliary objective only: reconstruct each cell's visible clue from the activity the reader sees."""
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Conv2d(10,64,3,padding=1),nn.Tanh(),nn.Conv2d(64,10,1))
    def loss(self,activity,x):
        grid=x.reshape(-1,16,16,10);mask=grid.sum(-1)>0;target=grid.argmax(-1)[mask];logits=self.net(activity).permute(0,2,3,1)[mask]
        clue=target>0;return F.cross_entropy(logits,target),float((logits.argmax(1)[clue]==target[clue]).float().mean())
