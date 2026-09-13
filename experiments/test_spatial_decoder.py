"""Spatial access contract: the intervention changes the receptive field."""
import unittest
import torch
from experiments.spatial_decoder import ActivityHead

class DecoderContract(unittest.TestCase):
 def test_receptive_field(self):
  torch.set_num_threads(1)
  for spatial in (False,True):
   head=ActivityHead(spatial)
   for p in head.parameters():torch.nn.init.constant_(p,.01)
   a=torch.zeros(1,10,16,16,requires_grad=True);ctx=torch.tensor([[5/16,3/25]])
   head(a,ctx)[0,8*16+8].backward();g=a.grad.abs().sum(1)[0]
   self.assertGreater(float(g[8,8]),0)
   if spatial:self.assertGreater(float(g[8,10]),0)
   else:self.assertEqual(float(g[8,10]),0)
   self.assertEqual(float(g[8,11]),0)
 def test_capacity_control(self):
  counts=[sum(p.numel() for p in ActivityHead(s).parameters()) for s in (False,True)]
  self.assertLess(abs(counts[0]/counts[1]-1),.01)
 def test_head_checkpoint_roundtrip(self):
  torch.manual_seed(1);m=ActivityHead(True);x=torch.randn(2,10,16,16);ctx=torch.randn(2,2)
  copy=ActivityHead(True);copy.load_state_dict(m.state_dict())
  torch.testing.assert_close(m(x,ctx),copy(x,ctx),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
