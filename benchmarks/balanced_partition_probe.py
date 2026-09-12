"""Paired row-vs-nonzero partition timings; same full graph and gradients."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from experiments.backbone import Backbone,encode_visible
from experiments.train import visible
from minesweeper import Minesweeper
b=Backbone(sparse.load_npz('data/processed/graph.npz'),'runs/curriculum-001/initial-expanded.npz',workers=4)
original=b.parts;balanced={}
for name,m in [('forward',b.graph),('backward',b.transpose)]:
 edges=np.searchsorted(m.indptr,np.linspace(0,m.nnz,5));edges[0]=0;edges[-1]=m.shape[0]
 balanced[name]=[sparse.csr_matrix((m.data[m.indptr[a]:m.indptr[z]],m.indices[m.indptr[a]:m.indptr[z]],m.indptr[a:z+1]-m.indptr[a]),shape=(z-a,m.shape[1]),copy=False) for a,z in zip(edges[:-1],edges[1:])]
x,_=encode_visible([visible(Minesweeper(77+i,16,40)) for i in range(16)]);df=np.ones((16,len(b.output_index)),np.float32)/16
rows=[];reference=None
for mode in ['rows','balanced','balanced','rows','rows','balanced']:
 b.parts=original if mode=='rows' else balanced
 t=time.perf_counter();f,c=b.forward(x);dg=b.backward(c,df);seconds=time.perf_counter()-t
 if reference is None:reference=(f.copy(),dg.copy())
 np.testing.assert_array_equal(f,reference[0]);np.testing.assert_array_equal(dg,reference[1])
 rows.append({'mode':mode,'seconds':seconds,'exact_equal':True})
b.close();out={'measurements':rows,'median_speedup':float(np.median([r['seconds'] for r in rows if r['mode']=='rows'])/np.median([r['seconds'] for r in rows if r['mode']=='balanced']))}
Path('benchmarks/balanced_partition_result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
