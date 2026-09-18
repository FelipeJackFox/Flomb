"""Publicly certified mine/safe auxiliary labels; uncertain cells have no target."""
import torch
from torch.nn import functional as F
from experiments.diagnose_mine_choices import info_from

def mine_labels(rows):
 result=torch.zeros(len(rows),256,dtype=torch.bool)
 for i,row in enumerate(rows):
  info=info_from(row)
  for cell in info.mines:result[i,cell//row['size']*16+cell%row['size']]=True
 return result

def outputs(head,aux,activity,context):
 hidden=head.net[:4](torch.cat([activity,context[:,:,None,None].expand(-1,-1,16,16)],1))
 return head.net[4](hidden).flatten(1),aux(hidden).flatten(1)

def certified_loss(logits,safe,mines):
 if (safe&mines).any():raise ValueError('Contradictory certificates')
 parts=[]
 if safe.any():parts.append(F.softplus(logits[safe]).mean())
 if mines.any():parts.append(F.softplus(-logits[mines]).mean())
 return torch.stack(parts).mean() if parts else logits.sum()*0
