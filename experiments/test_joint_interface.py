import unittest
import numpy as np
import torch
from scipy import sparse
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead,activity_map
from experiments.joint_interface import JointInterface

class JointTest(unittest.TestCase):
 def test_sparse_encoder_gradient_matches_dense_graph(self):
  torch.set_num_threads(1);torch.manual_seed(917);rng=np.random.default_rng(917)
  graph=sparse.csr_matrix(rng.normal(0,.1,(12,12)).astype(np.float32))
  mapping=dict(input_index=np.arange(9),input_xy=rng.random((9,2)).astype(np.float32),input_channel=np.arange(9)%3,outputs={})
  for size in (5,7):mapping['outputs'][size]=(rng.integers(12,size=(size*size,10,4)),np.full((size*size,10,4),.25,np.float32))
  op=PlasticOperator(graph,workers=2)
  try:
   brain=RetinaPolicy(op,mapping,True);model=JointInterface(brain,ActivityHead(True))
   x=torch.zeros(2,16,16,10);x[0,:5,:5,0]=1;x[1,:7,:7,0]=1;x=x.flatten(1);c=torch.tensor([[5/16,3/256],[7/16,7/256]])
   expected=activity_map(brain,x,c);actual=model.activity(x,c)
   torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
   weight=torch.randn_like(actual);loss=(actual*weight).sum();loss.backward()
   grads=[p.grad.clone() for p in brain.encoder.parameters()]
   self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
   self.assertIsNone(brain.edge_log_gain.grad);self.assertIsNone(brain.node_log_gain.grad)
   model.zero_grad();dense=model.activity(x,c,dense=torch.from_numpy(graph.toarray()))
   (dense*weight).sum().backward()
   for expected,p in zip(grads,brain.encoder.parameters()):torch.testing.assert_close(p.grad,expected,rtol=1e-4,atol=1e-5)
  finally:op.close()
if __name__=='__main__':unittest.main()
