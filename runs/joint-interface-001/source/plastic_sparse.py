"""Full-edge plasticity without allocating a dense N*N gradient or E*batch tensor."""
import ctypes
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scipy import sparse


def kernel():
    source=Path(__file__).with_name('plastic_kernel.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=Path('benchmarks/native');folder.mkdir(parents=True,exist_ok=True)
    library=folder/f'plastic-{digest}.dylib'
    if not library.exists():
        tmp=library.with_suffix('.tmp.dylib')
        subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
        tmp.replace(library)
    lib=ctypes.CDLL(str(library.resolve()))
    for name in ('edge_grad_f32','edge_grad_f64'):
        fn=getattr(lib,name)
        fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3
        fn.restype=None
    lib.rewire.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64,ctypes.c_int64,ctypes.c_uint64]
    lib.rewire.restype=ctypes.c_int64
    return lib


class PlasticOperator:
    def __init__(self,graph,workers=4):
        graph=graph.tocsr(copy=True);graph.sort_indices()
        if not graph.has_canonical_format or graph.nnz>=2**31:
            raise ValueError('Canonical int32 CSR required')
        self.shape=graph.shape
        self.ptr=graph.indptr.astype(np.int32,copy=False)
        self.col=graph.indices.astype(np.int32,copy=False)
        self.base=graph.data.astype(np.float32,copy=True)
        # The transpose permutation is structural and constructed only once.
        order=sparse.csr_matrix((np.arange(graph.nnz,dtype=np.int32),self.col,self.ptr),shape=self.shape).T.tocsr()
        self.tptr,self.tcol,self.order=order.indptr,order.indices,order.data
        self.workers=workers
        self.pool=ThreadPoolExecutor(workers)
        self.lib=kernel()
        self.boundaries={}
        for name,ptr in [('forward',self.ptr),('backward',self.tptr)]:
            bounds=np.searchsorted(ptr,np.linspace(0,graph.nnz,workers+1))
            bounds[0]=0;bounds[-1]=self.shape[0]
            self.boundaries[name]=list(zip(bounds[:-1],bounds[1:]))
        self.bound=None

    def bind(self,values,force=False):
        address=values.__array_interface__['data'][0]
        if not force and self.bound is not None and address==self.bound.__array_interface__['data'][0]:
            return
        self.bound=values
        self.parts={}
        for name,ptr,col,data in [('forward',self.ptr,self.col,values),
                                  ('backward',self.tptr,self.tcol,values[self.order])]:
            blocks=[]
            for a,b in self.boundaries[name]:
                lo,hi=ptr[a],ptr[b]
                blocks.append(sparse.csr_matrix((data[lo:hi],col[lo:hi],ptr[a:b+1]-lo),
                                               shape=(b-a,self.shape[0]),copy=False))
            self.parts[name]=blocks

    def multiply(self,values,x,transpose=False):
        self.bind(values)
        return np.concatenate(list(self.pool.map(lambda part:part@x,self.parts['backward' if transpose else 'forward'])))

    def gradient(self,x,dy):
        x=np.ascontiguousarray(x);dy=np.ascontiguousarray(dy)
        out=np.empty(len(self.col),dtype=x.dtype)
        fn=self.lib.edge_grad_f64 if x.dtype==np.float64 else self.lib.edge_grad_f32
        def work(bounds):
            a,b=bounds
            fn(self.ptr.ctypes.data,self.col.ctypes.data,x.ctypes.data,dy.ctypes.data,
               out.ctypes.data,int(a),int(b),x.shape[1])
        list(self.pool.map(work,self.boundaries['forward']))
        return out

    def close(self):
        self.pool.shutdown(wait=True)


class PlasticMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,x,operator):
        if values.device.type!='cpu' or x.device.type!='cpu':
            raise ValueError('Native sparse bridge requires CPU')
        ctx.operator=operator
        ctx.save_for_backward(values,x)
        operator.bind(values.detach().numpy(),force=True)
        return torch.from_numpy(operator.multiply(values.detach().numpy(),x.detach().contiguous().numpy()))

    @staticmethod
    def backward(ctx,dy):
        values,x=ctx.saved_tensors
        op=ctx.operator
        op.bind(values.detach().numpy(),force=True)
        raw=dy.detach().contiguous().numpy()
        dw=torch.from_numpy(op.gradient(x.detach().numpy(),raw)) if ctx.needs_input_grad[0] else None
        dx=torch.from_numpy(op.multiply(values.detach().numpy(),raw,True)) if ctx.needs_input_grad[1] else None
        return dw,dx,None


def reconfigured(graph,signs,seed=20260926,attempt_factor=1):
    graph=graph.tocsr(copy=True);graph.sort_indices()
    ptr=graph.indptr.astype(np.int32,copy=False)
    col=graph.indices.astype(np.int32,copy=True)
    rows=np.repeat(np.arange(graph.shape[0],dtype=np.int32),np.diff(ptr))
    signs=np.ascontiguousarray(signs,dtype=np.float32)
    lib=kernel()
    accepted=lib.rewire(ptr.ctypes.data,col.ctypes.data,rows.ctypes.data,signs.ctypes.data,
                        len(col),int(len(col)*attempt_factor),seed)
    changed=int(np.count_nonzero(col!=graph.indices))
    assert np.array_equal(np.bincount(col,minlength=graph.shape[0]),np.bincount(graph.indices,minlength=graph.shape[0]))
    assert np.array_equal(signs[col],signs[graph.indices])
    result=sparse.csr_matrix((graph.data.copy(),col,ptr.copy()),shape=graph.shape)
    result.sort_indices()
    assert result.has_canonical_format and result.nnz==graph.nnz
    assert np.array_equal(result.indptr,graph.indptr)
    return result,dict(attempts=int(len(col)*attempt_factor),accepted=int(accepted),
                       changed_edge_slots=changed,fraction_changed_slots=changed/len(col),seed=seed,
                       preserves='in/out degree, incoming signed-weight multiset, transmitter sign per edge, existing self-edges; not weighted out-degree')
