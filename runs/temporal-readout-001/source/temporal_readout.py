"""Matched-width readouts of one or two propagation times."""
import torch
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo

class TemporalHead(ActivityHead):
 def __init__(self,spatial=True):
  super().__init__(spatial)
  self.net[0]=torch.nn.Conv2d(22,32,3,padding=1)

class TemporalMemo:
 def __init__(self,brain,combined):
  self.brain=brain;self.combined=combined;self.first=ActivityMemo(brain);self.second=ActivityMemo(brain)
 @torch.no_grad()
 def get(self,x,c):
  self.brain.cycles=1;a=self.first.get(x,c)
  if self.combined:
   self.brain.cycles=2;b=self.second.get(x,c)
  else:b=a
  return torch.cat([a,b],1)
