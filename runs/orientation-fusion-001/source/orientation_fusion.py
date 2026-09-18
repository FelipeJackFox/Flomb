"""Learned fusion of four aligned brain readings instead of averaging logits.

The frozen brain is not rotation-equivariant, so A(R^k x) differs from R^k A(x).
Undoing the rotation on each activity map gives four aligned views of one board;
turning the board by j only rotates that stack and rolls it cyclically:
S(R^j x)_k = R^j S(x)_{(k+j)%4}.
"""
import torch
from torch import nn

def rotate_sized(t,context,k):
 """rot90 of the NxN corner over the last two dims, padding untouched."""
 out=t.clone();sizes=(context[:,0]*16).round().long()
 for n in sorted(set(sizes.tolist())):
  ix=torch.nonzero(sizes==n).flatten()
  out[ix,...,:n,:n]=torch.rot90(t[ix][...,:n,:n],k,(-2,-1))
 return out

def aligned_stack(a4,context,j=None):
 """a4[b,m]=A(R^m x_b), shape (B,4,C,16,16) -> S(R^j x) as (B,4C,16,16)."""
 j=torch.zeros(len(a4),dtype=torch.long) if j is None else torch.as_tensor(j,dtype=torch.long)
 rows=torch.arange(len(a4))
 return torch.cat([rotate_sized(a4[rows,(k+j)%4],context,-k) for k in range(4)],1)

class FusionHead(nn.Module):
 def __init__(self,width=22):
  super().__init__()
  self.net=nn.Sequential(nn.Conv2d(42,width,3,padding=1),nn.Tanh(),nn.Conv2d(width,width,3,padding=1),nn.Tanh(),nn.Conv2d(width,1,1))
 def forward(self,stack,context):
  return self.net(torch.cat([stack,context[:,:,None,None].expand(-1,-1,16,16)],1)).flatten(1)

class FusionPolicy(nn.Module):
 """Four brain forwards per decision, the same nominal cost as RotationPolicy."""
 def __init__(self,memo,head):super().__init__();self.memo=memo;self.head=head
 def forward(self,x,c):
  grid=x.reshape(-1,16,16,10).permute(0,3,1,2)
  views=[self.memo.get(rotate_sized(grid,c,k).permute(0,2,3,1).reshape(-1,2560),c) for k in range(4)]
  return self.head(aligned_stack(torch.stack(views,1),c),c)
