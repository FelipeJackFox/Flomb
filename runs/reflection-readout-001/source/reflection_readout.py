"""Average all eight square symmetries using rotation consensus twice."""
import torch
from experiments.rotation_readout import RotationPolicy

def reflect_grid(grid,c):
 out=grid.clone()
 for i,n in enumerate((c[:,0]*16).round().long().tolist()):
  out[i,:n,:n]=grid[i,:n,:n].flip(1)
 return out

class ReflectionPolicy(torch.nn.Module):
 def __init__(self,policy):
  super().__init__();self.rotations=RotationPolicy(policy)
 def forward(self,x,c):
  a=self.rotations(x,c).reshape(-1,16,16)
  reflected=reflect_grid(x.reshape(-1,16,16,10),c).flatten(1)
  b=reflect_grid(self.rotations(reflected,c).reshape(-1,16,16),c)
  return ((a+b)/2).flatten(1)
