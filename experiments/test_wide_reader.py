import unittest,torch
from experiments.wide_reader import make_head
class WideReaderTest(unittest.TestCase):
 def test_same_parameters_and_exact_receptive_fields(self):
  torch.set_num_threads(1)
  for dilation,radius in [(1,2),(3,4)]:
   h=make_head(dilation)
   self.assertEqual(sum(p.numel() for p in h.parameters()),12769)
   for p in h.parameters():torch.nn.init.constant_(p,.01)
   x=torch.zeros(1,10,16,16,requires_grad=True);c=torch.zeros(1,2)
   h(x,c)[0,8*16+8].backward();mask=x.grad[0,0]!=0
   expected=torch.zeros(16,16,dtype=torch.bool);expected[8-radius:9+radius,8-radius:9+radius]=True
   self.assertTrue(torch.equal(mask,expected))
if __name__=='__main__':unittest.main()
