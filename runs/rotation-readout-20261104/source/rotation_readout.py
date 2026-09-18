"""Four board orientations with inverse-mapped logits, preserving padding."""
import torch

def rotate_grid(grid,context,k):
 out=grid.clone()
 for i,n in enumerate((context[:,0]*16).round().long().tolist()):
  out[i,:n,:n]=torch.rot90(grid[i,:n,:n],k,(0,1))
 return out

class RotationPolicy(torch.nn.Module):
 def __init__(self,policy):
  super().__init__();self.policy=policy
 def forward(self,x,c):
  grid=x.reshape(-1,16,16,10);values=[]
  for k in range(4):
   logits=self.policy(rotate_grid(grid,c,k).reshape(-1,2560),c).reshape(-1,16,16)
   values.append(rotate_grid(logits,c,-k))
  return torch.stack(values).mean(0).flatten(1)
