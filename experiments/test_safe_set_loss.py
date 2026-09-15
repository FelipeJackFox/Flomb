import unittest
import torch
from experiments.safe_set_loss import safe_set_loss
from experiments.expressive_models import equivalent_loss

class TestSafeSet(unittest.TestCase):
    def test_singleton_value_and_gradient(self):
        z=torch.tensor([[.5,-1.,2.,8.]],requires_grad=True);l=torch.tensor([[True,True,True,False]]);y=torch.tensor([[False,True,False,False]])
        a=safe_set_loss(z,l,y);b=equivalent_loss(z,l,y)
        torch.testing.assert_close(a,b)
        ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
        torch.testing.assert_close(ga,gb);self.assertEqual(ga[0,3].item(),0.)
    def test_same_total_safe_mass(self):
        z=torch.tensor([[.4,.4,.2],[.79,.01,.2]]).log();l=torch.ones(2,3,dtype=torch.bool);y=torch.tensor([[True,True,False]]*2)
        torch.testing.assert_close(safe_set_loss(z[:1],l[:1],y[:1]),safe_set_loss(z[1:],l[1:],y[1:]))
        self.assertGreater(equivalent_loss(z[1:],l[1:],y[1:]),equivalent_loss(z[:1],l[:1],y[:1]))
    def test_shift_and_all_safe(self):
        z=torch.tensor([[1.,-3.,8.]],dtype=torch.float64);l=torch.ones_like(z,dtype=torch.bool);y=torch.tensor([[True,False,True]])
        torch.testing.assert_close(safe_set_loss(z,l,y),safe_set_loss(z+10000,l,y))
        self.assertAlmostEqual(safe_set_loss(z,l,l).item(),0.)
    def test_invalid(self):
        z=torch.zeros(1,2);l=torch.tensor([[True,False]])
        for y in (torch.tensor([[False,False]]),torch.tensor([[False,True]])):
            with self.assertRaises(ValueError):safe_set_loss(z,l,y)

if __name__=='__main__':unittest.main()
