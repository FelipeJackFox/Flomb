import unittest
import numpy as np
import torch
from scipy import sparse
from experiments.plastic_sparse import PlasticOperator
from experiments.retina_policy import RetinaPolicy
from experiments.spatial_decoder import ActivityHead,activity_map
from experiments.synaptic_learning import SynapticBrain
from experiments.expressive_models import equivalent_loss

def toy(cycles=3):
    rng=np.random.default_rng(1);n=400
    graph=sparse.random(n,n,density=.05,random_state=2,data_rvs=lambda k:rng.normal(size=k)*.3,format='csr')
    outputs={}
    for size in (5,7):
        outputs[size]=(rng.integers(100,n,size=(size*size,10,4)).astype(np.int64),np.full((size*size,10,4),.25,np.float32))
    mapping=dict(input_index=np.arange(60,dtype=np.int64),input_xy=rng.random((60,2)).astype(np.float32),input_channel=rng.integers(3,size=60).astype(np.int64),outputs=outputs)
    op=PlasticOperator(graph,workers=2);torch.manual_seed(4);brain=RetinaPolicy(op,mapping,True,cycles=cycles);head=ActivityHead(True)
    return op,brain,head

def batch():
    torch.manual_seed(7);x=torch.zeros(3,16,16,10);c=torch.zeros(3,2);l=torch.zeros(3,256);y=torch.zeros(3,256)
    for i,n in enumerate((5,7,7)):
        x[i,:n,:n]=torch.nn.functional.one_hot(torch.randint(10,(n,n)),10).float();c[i,0]=n/16;c[i,1]=.1
        cells=[r*16+q for r in range(n) for q in range(n)];l[i,cells]=1;y[i,cells[:3]]=1
    return x.reshape(3,-1),l.bool(),c,y.bool()

class Tests(unittest.TestCase):
    def test_forward_matches_inference_path(self):
        op,brain,head=toy()
        try:
            x,l,c,y=batch();model=SynapticBrain(brain,head)
            torch.testing.assert_close(model.activity(x,c).detach(),activity_map(brain,x,c),rtol=1e-5,atol=1e-6)
        finally:op.close()
    def test_only_synapses_receive_gradient_and_match_finite_differences(self):
        op,brain,head=toy()
        try:
            x,l,c,y=batch();model=SynapticBrain(brain,head).double();brain.base=brain.base.double()
            x,c=x.double(),c.double()
            loss=lambda:equivalent_loss(model(x,c),l,y)
            model.zero_grad();loss().backward()
            self.assertTrue(all(p.grad is None for p in list(head.parameters())+list(brain.encoder.parameters())))
            grad=brain.edge_log_gain.grad.clone();node=brain.node_log_gain.grad.clone()
            for param,g in ((brain.edge_log_gain,grad),(brain.node_log_gain,node)):
                for index in torch.topk(g.abs(),3).indices.tolist():
                    with torch.no_grad():
                        param[index]+=1e-5;up=float(loss());param[index]-=2e-5;down=float(loss());param[index]+=1e-5
                    self.assertAlmostEqual((up-down)/2e-5,float(g[index]),delta=1e-6+1e-4*abs(float(g[index])))
        finally:op.close()
    def test_learned_gains_change_the_inference_path(self):
        op,brain,head=toy()
        try:
            x,l,c,y=batch();model=SynapticBrain(brain,head);before=activity_map(brain,x,c)
            opt=torch.optim.Adam(model.synapses(),lr=.05);start=float(equivalent_loss(model(x,c),l,y))
            for _ in range(30):opt.zero_grad();equivalent_loss(model(x,c),l,y).backward();opt.step()
            self.assertLess(float(equivalent_loss(model(x,c),l,y)),start)
            self.assertGreater(float((activity_map(brain,x,c)-before).abs().max()),0)
        finally:op.close()

if __name__=='__main__':unittest.main()
