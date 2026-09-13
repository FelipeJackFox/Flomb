import copy
import contextlib
import hashlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy import sparse
from experiments.plastic_sparse import PlasticOperator,PlasticMessage,reconfigured
from experiments.retina_policy import RetinaPolicy


def tiny():
    rng=np.random.default_rng(26)
    graph=sparse.csr_matrix(rng.normal(0,.12,(12,12)).astype(np.float32))
    mapping=dict(input_index=np.arange(3,dtype=np.int64),input_xy=np.array([[0,0],[.5,.5],[1,1]],np.float32),
                 input_channel=np.arange(3,dtype=np.int64),outputs={})
    for size in (5,7):
        idx=rng.integers(3,12,(size*size,10,4),dtype=np.int64)
        mapping['outputs'][size]=(idx,np.full(idx.shape,.25,np.float32))
    return graph,mapping


class Tests(unittest.TestCase):
    def test_edge_and_state_gradients_numerical_and_dense(self):
        torch.set_num_threads(1)
        graph,_=tiny();op=PlasticOperator(graph,workers=2)
        try:
            values=torch.tensor(graph.data,dtype=torch.float64,requires_grad=True)
            x=torch.randn(12,3,dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda w,h:PlasticMessage.apply(w,h,op),(values,x)))
            out=PlasticMessage.apply(values,x,op)
            matrix=torch.zeros(12,12,dtype=torch.float64)
            rows=np.repeat(np.arange(12),np.diff(graph.indptr))
            matrix[rows,graph.indices]=values
            torch.testing.assert_close(out,matrix@x)
            a=torch.autograd.grad(out.square().sum(),(values,x),retain_graph=True)
            b=torch.autograd.grad((matrix@x).square().sum(),(values,x))
            for first,second in zip(a,b):torch.testing.assert_close(first,second)
        finally:op.close()

    def test_independent_edge_updates_and_zero_ablation(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        graph,mapping=tiny();op=PlasticOperator(graph,workers=2)
        try:
            model=RetinaPolicy(op,mapping)
            x=torch.zeros(2,16,16,10);x[:,:7,:7,0]=1;x[:,2,3]=0;x[:,2,3,2]=1
            x=x.flatten(1);context=torch.tensor([[5/16,3/25],[7/16,7/49]])
            loss=model(x,context).square().sum();loss.backward()
            self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0)
            self.assertGreater(float(model.edge_log_gain.grad.abs().sum()),0)
            # Two outputs of the same presynaptic neuron can receive different gradients.
            g=model.edge_log_gain.grad.detach().numpy()[op.col==0]
            self.assertGreater(float(np.ptp(g)),1e-9)
            z=model(x,context,ablation='zero')
            for i,size in enumerate((5,7)):
                board=z[i].reshape(16,16)[:size,:size]
                torch.testing.assert_close(board,board[0,0].expand_as(board))
        finally:op.close()

    def test_rewiring_preserves_degrees_signs_and_input(self):
        rng=np.random.default_rng(12)
        graph=sparse.random(80,80,density=.12,random_state=rng,format='csr',dtype=np.float32)
        signs=np.where(np.arange(80)%3,1,-1).astype(np.float32)
        graph.data*=signs[graph.indices]
        original=graph.copy()
        rewired,report=reconfigured(graph,signs,attempt_factor=3)
        self.assertGreater(report['fraction_changed_slots'],.2)
        np.testing.assert_array_equal(graph.indices,original.indices)
        np.testing.assert_array_equal(graph.data,original.data)
        np.testing.assert_array_equal(np.diff(graph.indptr),np.diff(rewired.indptr))
        np.testing.assert_array_equal(np.bincount(graph.indices,minlength=80),np.bincount(rewired.indices,minlength=80))
        np.testing.assert_array_equal(np.sign(rewired.data),signs[rewired.indices])

    def test_resume_optimizer_full_edge_state(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        graph,mapping=tiny();op=PlasticOperator(graph,workers=2)
        try:
            model=RetinaPolicy(op,mapping);opt=torch.optim.Adam(model.parameters(),lr=.001)
            x=torch.randn(2,2560);ctx=torch.tensor([[5/16,3/25]]*2)
            def step(m,o):
                o.zero_grad();m(x,ctx).square().sum().backward();o.step()
            step(model,opt)
            state=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
            other=RetinaPolicy(op,mapping);other.load_state_dict(state)
            opt2=torch.optim.Adam(other.parameters(),lr=.001);opt2.load_state_dict(adam)
            step(model,opt);step(other,opt2)
            for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        finally:op.close()

    def test_structural_cli_resume_exact(self):
        from experiments.capacity_probe import build_dataset
        from experiments.structural_train import main
        torch.set_num_threads(1)
        graph,mapping=tiny()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'dataset.pkl').write_bytes(pickle.dumps(build_dataset(2,2,2)))
            (root/'mapping.pkl').write_bytes(pickle.dumps(mapping))
            protected=hashlib.sha256(Path('runs/hybrid-001/checkpoint.pkl').read_bytes()).hexdigest()
            (root/'prepared.json').write_text(json.dumps(dict(hashes={},protected_sha256=protected)))
            def invoke(name,extra):
                argv=['structural_train','--root',str(root),'--out',str(root/name),
                      '--variants','retina_plastic','--updates','3','--batch','4',*extra]
                with patch.object(sys,'argv',argv),patch('experiments.structural_train.sparse.load_npz',return_value=graph),contextlib.redirect_stdout(io.StringIO()):
                    main()
            invoke('continuous',[]);invoke('resumed',['--stop-after','1']);invoke('resumed',['--resume'])
            a=torch.load(root/'continuous/retina_plastic-20260926.pt',weights_only=False)
            b=torch.load(root/'resumed/retina_plastic-20260926.pt',weights_only=False)
            self.assertEqual(a['rng'],b['rng'])
            self.assertEqual(a['step'],b['step'])
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            for i,state in a['optimizer']['state'].items():
                for k,v in state.items():torch.testing.assert_close(v,b['optimizer']['state'][i][k],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
