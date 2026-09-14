import unittest
import numpy as np
import torch
from experiments.prioritize_mine_errors import mistakes
class FixedHead(torch.nn.Module):
 def forward(self,a,c):return a
class TestPriority(unittest.TestCase):
 def test_only_legal_selected_mines_enter_hard_pool(self):
  logits=torch.tensor([[0.,3.,4.],[4.,3.,0.],[0.,4.,3.]])
  legal=torch.tensor([[True,True,False],[True,True,True],[True,True,True]])
  safe=torch.tensor([[True,False,False],[True,False,False],[True,False,False]])
  mines=torch.tensor([[False,True,False],[False,True,False],[False,False,True]])
  hard,hits=mistakes(FixedHead(),(logits,legal,torch.zeros(3,2),safe),mines)
  np.testing.assert_array_equal(hard,[0]);self.assertEqual(hits,1)
if __name__=='__main__':unittest.main()
