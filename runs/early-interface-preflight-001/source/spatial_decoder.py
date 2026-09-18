"""Frozen connectome feature maps and a neural-activity-only spatial decoder."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@torch.no_grad()
def activity_map(brain,x,context):
 b=len(x);sizes=(context[:,0]*16).round().long()
 mapped_sizes=sorted(set(sizes.tolist()))
 for size in mapped_sizes:
  if not 2<=size<=16 or not hasattr(brain,f'out_{size}') or not hasattr(brain,f'weight_{size}'):
   raise ValueError(f'Missing output mapping for {size}x{size}; refusing silent zero activity')
 grid=x.reshape(b,16,16,10).permute(0,3,1,2)
 encoded=brain.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],1))
 encoded*=grid.sum(1,keepdim=True)!=0
 sg=2*brain.input_xy[None]*(sizes[:,None,None]-1)/15-1
 samples=F.grid_sample(encoded,sg[:,:,None],align_corners=True).squeeze(-1)
 sensory=samples[:,brain.input_channel,torch.arange(len(brain.input_channel))]
 drive=x.new_zeros(brain.operator.shape[0],b).index_copy(0,brain.input_index,sensory.T)
 state=torch.tanh(drive);values=brain.base*(brain.edge_log_gain.clamp(-2,2).exp() if brain.plastic else 1)
 gain=brain.node_log_gain.clamp(-1,1).exp()[:,None]
 brain.operator.bind(values.numpy(),force=True)
 for _ in range(brain.cycles):state=torch.tanh(torch.from_numpy(brain.operator.multiply(values.numpy(),(gain*state).numpy()))+.15*drive)
 result=x.new_zeros(b,10,16,16)
 for size in mapped_sizes:
  selected=torch.nonzero(sizes==size).flatten()
  if not len(selected):continue
  ix=getattr(brain,f'out_{size}');wt=getattr(brain,f'weight_{size}')
  pooled=(state[ix][:,:,:,selected]*wt[:,:,:,None]).sum(2).permute(2,1,0)
  result[selected,:,:size,:size]=pooled.reshape(len(selected),10,size,size)*10
 return result

class ActivityHead(nn.Module):
 def __init__(self,spatial):
  super().__init__();width,k=(32,3) if spatial else (106,1)
  self.net=nn.Sequential(nn.Conv2d(12,width,k,padding=k//2),nn.Tanh(),nn.Conv2d(width,width,k,padding=k//2),nn.Tanh(),nn.Conv2d(width,1,1))
 def forward(self,activity,context):
  # Context is board dimensions/mine count, never the visible clues.
  return self.net(torch.cat([activity,context[:,:,None,None].expand(-1,-1,16,16)],1)).flatten(1)

class FrozenPolicy(nn.Module):
 def __init__(self,brain,head):super().__init__();self.brain=brain;self.head=head
 def forward(self,x,context):return self.head(activity_map(self.brain,x,context),context)
