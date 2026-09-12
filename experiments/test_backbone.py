import tempfile,unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from experiments.backbone import Backbone,encode_visible
from experiments.train import supervised,teacher
from minesweeper import Minesweeper

class Tests(unittest.TestCase):
 def test_batched_forward_and_gradient(self):
  rng=np.random.default_rng(12);g=sparse.csr_matrix(rng.normal(0,.1,(12,12)).astype(np.float32))
  old=BrainPolicy(g,inputs=250,readouts=8)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'init.npz';old.save(f);b=Backbone(g,f);parallel=Backbone(g,f,workers=3)
  x=rng.normal(0,1,(3,250)).astype(np.float32);features,c=b.forward(x)
  expected=np.array([old.forward(y,np.ones(25,bool),True)[1][1] for y in x])
  np.testing.assert_allclose(features,expected,rtol=2e-5,atol=2e-6)
  upstream=rng.normal(size=features.shape).astype(np.float32);grad=b.backward(c,upstream)
  pf,pc=parallel.forward(x);np.testing.assert_array_equal(pf,features);np.testing.assert_array_equal(parallel.backward(pc,upstream),grad);parallel.close()
  for i in [0,4,11]:
   orig=b.log_gain[i];eps=.002
   b.log_gain[i]=orig+eps;plus=np.sum(b.forward(x)[0]*upstream)
   b.log_gain[i]=orig-eps;minus=np.sum(b.forward(x)[0]*upstream)
   b.log_gain[i]=orig
   self.assertAlmostEqual(float(grad[i]),float((plus-minus)/(2*eps)),delta=.002)
 def test_padding_and_teacher_visible_only(self):
  env=Minesweeper(4,5,3);v=np.full((16,16),-2,np.int8);v[:5,:5]=env.visible.reshape(5,5)
  x,mask=encode_visible([v.ravel()]);self.assertEqual(x.reshape(16,16,10)[7,7].sum(),0)
  np.testing.assert_array_equal(mask[0].reshape(16,16)[:5,:5],env.legal_mask().reshape(5,5))
  labels,w=teacher(env);env._mines[:]=~env._mines # teacher must ignore all hidden truth
  other,ow=teacher(env);np.testing.assert_array_equal(labels,other);self.assertEqual(w,ow)
 def test_supervised_gradient(self):
  z=np.array([[[.2,.4],[.1,-.2],[.5,.3]]],np.float32);mask=np.array([[1,0,1]],bool);labels=np.array([[0,0,1]],bool)
  loss,d=supervised(z,mask,labels,np.ones(1),np.ones(1));self.assertEqual(d[0,1].sum(),0)
  eps=.001;zp=z.copy();zm=z.copy();zp[0,0,0]+=eps;zm[0,0,0]-=eps
  numeric=(supervised(zp,mask,labels,np.ones(1),np.ones(1))[0]-supervised(zm,mask,labels,np.ones(1),np.ones(1))[0])/(2*eps)
  self.assertAlmostEqual(float(d[0,0,0]),numeric,delta=.0001)
if __name__=='__main__':unittest.main()
