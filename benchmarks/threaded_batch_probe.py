"""Compare row-partitioned SpMM without changing trainer source."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse
from experiments.backbone import Backbone,encode_visible
from experiments.train import visible
from minesweeper import Minesweeper
class Partition:
 def __init__(self,matrix,pool,workers):
  self.pool=pool;edges=np.linspace(0,matrix.shape[0],workers+1,dtype=int);self.parts=[]
  for a,b in zip(edges[:-1],edges[1:]):
   lo,hi=matrix.indptr[a],matrix.indptr[b]
   self.parts.append(sparse.csr_matrix((matrix.data[lo:hi],matrix.indices[lo:hi],matrix.indptr[a:b+1]-lo),shape=(b-a,matrix.shape[1]),copy=False))
 def __matmul__(self,x):return np.concatenate(list(self.pool.map(lambda p:p@x,self.parts)))
b=Backbone(sparse.load_npz('data/processed/graph.npz'),'runs/curriculum-001/initial-expanded.npz');g,t=b.graph,b.transpose
x,_=encode_visible([visible(Minesweeper(77+i,16,40)) for i in range(16)]);df=np.ones((16,len(b.output_index)),np.float32)/16
results=[];ref=None
for workers in [1,2,4]:
 with ThreadPoolExecutor(workers) as pool:
  b.graph=g if workers==1 else Partition(g,pool,workers);b.transpose=t if workers==1 else Partition(t,pool,workers)
  start=time.perf_counter();f,c=b.forward(x);dg=b.backward(c,df);elapsed=time.perf_counter()-start
  if ref is None:ref=(f.copy(),dg.copy())
  else:
   np.testing.assert_array_equal(f,ref[0]);np.testing.assert_array_equal(dg,ref[1])
  results.append({'workers':workers,'seconds':elapsed,'batch':16,'exact_equal':True})
Path('benchmarks/threaded_batch_probe_result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
