"""Batched full-connectome rate backbone; no frozen/stale feature approximation."""
from concurrent.futures import ThreadPoolExecutor
from scipy import sparse
import numpy as np

class Backbone:
    def __init__(self, graph, checkpoint, lr=1e-3, workers=1):
        self.graph=graph;self.transpose=graph.T.tocsr();self.lr=lr
        self.workers=workers;self.pool=None;self.parts={}
        if workers<1:raise ValueError("workers must be positive")
        if workers>1:
            self.pool=ThreadPoolExecutor(workers)
            for name,matrix in [("forward",self.graph),("backward",self.transpose)]:
                edges=np.linspace(0,matrix.shape[0],min(workers,matrix.shape[0])+1,dtype=int);parts=[]
                for a,b in zip(edges[:-1],edges[1:]):
                    lo,hi=matrix.indptr[a],matrix.indptr[b]
                    parts.append(sparse.csr_matrix((matrix.data[lo:hi],matrix.indices[lo:hi],matrix.indptr[a:b+1]-lo),shape=(b-a,matrix.shape[1]),copy=False))
                self.parts[name]=parts
        with np.load(checkpoint) as z:
            for key in ('log_gain','input_index','input_sign','extra_input_index','extra_input_sign','output_index'):
                setattr(self,key,z[key].copy())
            self.cycles=int(z['cycles']);self.feature_scale=float(z['feature_scale'])
        self.m=np.zeros_like(self.log_gain);self.v=self.m.copy();self.updates=0
    def multiply(self,x,backward=False):
        if self.pool is None:return (self.transpose if backward else self.graph)@x
        return np.concatenate(list(self.pool.map(lambda part:part@x,self.parts["backward" if backward else "forward"])))
    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True);self.pool=None
    def forward(self, observations, cache=True):
        # Neurons x batch: one sparse matrix-matrix product reuses each graph row.
        x=np.asarray(observations,dtype=np.float32)
        drive=np.zeros((len(self.log_gain),len(x)),np.float32)
        for idx,sign in [(self.input_index,self.input_sign),(self.extra_input_index,self.extra_input_sign)]:
            for j in range(idx.shape[1]):drive+=x[:,idx[:,j]].T*sign[:,j,None]
        h=np.tanh(drive);states=[h] if cache else None;gain=np.exp(self.log_gain)[:,None]
        for _ in range(self.cycles):
            h=np.tanh(self.multiply(gain*h))
            if cache:states.append(h)
        return h[self.output_index].T*self.feature_scale, (states,gain) if cache else None
    def backward(self,cache,df):
        states,gain=cache;dh=np.zeros_like(states[-1]);dh[self.output_index]=df.T*self.feature_scale
        dg=np.zeros_like(self.log_gain)
        for k in range(self.cycles,0,-1):
            routed=self.multiply(dh*(1-states[k]**2),True)
            dg+=(routed*states[k-1]*gain).sum(axis=1)
            dh=routed*gain
        return dg
    def apply(self,g):
        if not np.isfinite(g).all():raise FloatingPointError('backbone gradient')
        norm=float(np.linalg.norm(g));g=g*min(1.,5/max(norm,1e-12));self.updates+=1
        self.m=.9*self.m+.1*g;self.v=.999*self.v+.001*g*g
        self.log_gain-=self.lr*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        np.clip(self.log_gain,-1,1,out=self.log_gain)
        return norm
    def state_dict(self):
        return {k:getattr(self,k).copy() if isinstance(getattr(self,k),np.ndarray) else getattr(self,k) for k in ('log_gain','m','v','updates','lr')}
    def load_state_dict(self,s):
        for k,v in s.items():setattr(self,k,v.copy() if isinstance(v,np.ndarray) else v)
    def target(self):
        import copy
        result=copy.copy(self);result.log_gain=self.log_gain.copy();return result

def encode_visible(boards):
    # Stored int8: -2 padding, -1 hidden, 0..8 clues; materialize one-hot per batch.
    cells=np.asarray(boards,dtype=np.int8);x=np.zeros((*cells.shape,10),np.float32)
    rows,cols=np.nonzero(cells>=-1);x[rows,cols,cells[rows,cols]+1]=1
    return x.reshape(len(cells),-1),cells==-1
