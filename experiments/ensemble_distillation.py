"""Masked soft-target distillation from a fixed four-orientation teacher."""
import torch
from experiments.capacity_probe import arrays
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.rotation_readout import RotationPolicy

def distillation_loss(logits,teacher,legal,temperature=2.):
 assert legal.bool().any(1).all()
 p=torch.softmax((teacher/temperature).masked_fill(~legal.bool(),-1e9),1)
 logq=torch.log_softmax((logits/temperature).masked_fill(~legal.bool(),-1e9),1)
 return torch.nn.functional.kl_div(logq,p,reduction='batchmean')*temperature**2

@torch.no_grad()
def teacher_targets(brain,head,rows):
 chunks=[]
 for start in range(0,len(rows),16):
  x,l,c,y=arrays(rows[start:start+16])
  policy=RotationPolicy(MemoPolicy(ActivityMemo(brain),head))
  chunks.append(policy(x,c))
 return torch.cat(chunks)
