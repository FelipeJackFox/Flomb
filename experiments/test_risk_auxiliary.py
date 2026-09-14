import unittest,torch
from experiments.risk_auxiliary import certified_loss,outputs
from experiments.spatial_decoder import ActivityHead
class RiskTest(unittest.TestCase):
 def test_unknown_cells_have_no_gradient_and_certified_signs(self):
  z=torch.zeros(1,4,requires_grad=True);safe=torch.tensor([[True,False,False,False]]);mines=torch.tensor([[False,True,False,False]])
  certified_loss(z,safe,mines).backward();self.assertGreater(z.grad[0,0],0);self.assertLess(z.grad[0,1],0);self.assertEqual(z.grad[0,2:].abs().sum(),0)
 def test_auxiliary_does_not_change_policy_forward(self):
  h=ActivityHead(True);aux=torch.nn.Conv2d(32,1,1);x=torch.randn(2,10,16,16);c=torch.randn(2,2)
  policy,_=outputs(h,aux,x,c);torch.testing.assert_close(policy,h(x,c),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
