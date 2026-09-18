"""Aligned rotation of observations, legal actions and certified-safe labels."""
import torch
from experiments.rotation_readout import rotate_grid
from experiments.capacity_probe import arrays
from experiments.validate_spatial_decoder import ActivityMemo

def rotate_arrays(x,l,c,y,k):
 return (rotate_grid(x.reshape(-1,16,16,10),c,k).flatten(1),rotate_grid(l.reshape(-1,16,16),c,k).flatten(1),c,rotate_grid(y.reshape(-1,16,16),c,k).flatten(1))

def rotation_features(brain,rows):
 chunks=[]
 for start in range(0,len(rows),16):
  original=arrays(rows[start:start+16]);variants=[]
  for k in range(4):
   x,l,c,y=rotate_arrays(*original,k)
   assert (y & ~l.bool()).sum()==0
   variants.append((ActivityMemo(brain).get(x,c),l,c,y))
  chunks.append(tuple(torch.stack([v[j] for v in variants],1).flatten(0,1) for j in range(4)))
 return tuple(torch.cat([p[j] for p in chunks]) for j in range(4))
