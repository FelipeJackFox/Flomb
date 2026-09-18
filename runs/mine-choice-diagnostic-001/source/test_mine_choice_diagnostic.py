import unittest
import numpy as np
import torch
from experiments.diagnose_mine_choices import proof,info_from
from experiments.expressive_models import equivalent_loss
class DiagnosticTests(unittest.TestCase):
 def test_single_clue_certificate_uses_visible_board(self):
  b=np.full((16,16),-2,np.int8);b[:5,:5]=-1;b[2,2]=8
  r=dict(board=b.flatten(),size=5,mines=8);info=info_from(r)
  kind,w=proof(r['board'],5,8,7,info)
  self.assertEqual(kind,'single_clue');self.assertEqual(w[0]['number'],8);self.assertEqual(len(w[0]['hidden_neighbors']),8)
  self.assertEqual(proof(r['board'],5,8,0,info)[0],'not_proven_mine')
 def test_loss_does_not_distinguish_two_non_safe_targets(self):
  legal=torch.ones((1,3),dtype=torch.bool);labels=torch.tensor([[True,False,False]])
  a=torch.tensor([[1.,3.,2.]],requires_grad=True);b=torch.tensor([[1.,2.,3.]],requires_grad=True)
  torch.testing.assert_close(equivalent_loss(a,legal,labels),equivalent_loss(b,legal,labels),atol=1e-6,rtol=1e-6)
  a2=torch.tensor([[1.,2.,2.]],requires_grad=True);equivalent_loss(a2,legal,labels).backward();self.assertEqual(float(a2.grad[0,1]),float(a2.grad[0,2]))
if __name__=='__main__':unittest.main()
