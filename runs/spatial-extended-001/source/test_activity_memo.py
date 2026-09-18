import unittest
from unittest.mock import patch
import torch
from experiments.validate_spatial_decoder import ActivityMemo

class MemoTest(unittest.TestCase):
 def test_public_observation_and_context_are_both_keys(self):
  calls=[]
  def fake(brain,x,c):
   calls.append(len(x));return (x[:,0]+c[:,0])[:,None,None,None].expand(-1,10,16,16).clone()
  memo=ActivityMemo(None)
  x=torch.tensor([[1.,0.],[2.,0.],[1.,0.]])
  c=torch.tensor([[.3,0.],[.3,0.],[.3,0.]])
  with patch('experiments.validate_spatial_decoder.activity_map',fake):
   a=memo.get(x,c);self.assertEqual(calls,[2])
   torch.testing.assert_close(memo.get(x.flip(0),c.flip(0)),a.flip(0),rtol=0,atol=0)
   self.assertEqual(calls,[2])
   different=memo.get(x[:1],c[:1]+1)
   self.assertEqual(calls,[2,1]);self.assertFalse(torch.equal(different,a[:1]))
   a.zero_();self.assertGreater(float(memo.get(x[:1],c[:1]).sum()),0)

if __name__=='__main__':unittest.main()
