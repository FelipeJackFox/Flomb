"""Read-only sparse compute benchmark; never updates the live training state."""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper
from curriculum import encode

def timing(fn, repeat=12):
 vals=[]
 for _ in range(repeat):
  start=time.perf_counter(); fn(); vals.append(time.perf_counter()-start)
 return float(np.median(vals))

graph=sparse.load_npz('data/processed/graph.npz')
p=BrainPolicy(graph)
state=json.loads(Path('runs/curriculum-001/state.json').read_text())
p.load(Path('runs/curriculum-001')/state['checkpoint'])
obs,mask=encode(Minesweeper(1984,16,40))
x=np.random.default_rng(0).normal(size=graph.shape[0]).astype(np.float32)
expected=graph@x
result={'checkpoint':state['checkpoint'],'note':'Measured alongside ongoing training; not an end-to-end training speed guarantee.'}
serial=timing(lambda:graph@x,30)
result['sparse_serial_ms']=serial*1000
for workers in (2,4):
 edges=np.linspace(0,graph.shape[0],workers+1,dtype=int)
 parts=[graph[edges[i]:edges[i+1]] for i in range(workers)]
 with ThreadPoolExecutor(max_workers=workers) as pool:
  def run(): return np.concatenate(list(pool.map(lambda a:a@x,parts)))
  np.testing.assert_allclose(run(),expected,rtol=1e-6,atol=1e-6)
  elapsed=timing(run,30)
 result[f'sparse_{workers}_workers_ms']=elapsed*1000
 result[f'sparse_{workers}_workers_speedup']=serial/elapsed
 del parts
probs,cache=p.forward(obs,mask,True)
d=-probs.copy();d[int(np.flatnonzero(mask)[0])]+=1

def cached():
 probs,cache=p.forward(obs,mask,True)
 return p.backward(cache,d)
def recomputed():
 p.forward(obs,mask)
 probs,cache=p.forward(obs,mask,True)
 return p.backward(cache,d)
for a,b in zip(cached(),recomputed()):np.testing.assert_array_equal(a,b)
a=timing(cached);b=timing(recomputed)
result.update(cached_forward_backward_ms=a*1000,recomputed_forward_backward_ms=b*1000,cache_speedup=b/a,cache_bytes_per_click=sum(v.nbytes for v in cache[0])+cache[1].nbytes+cache[2].nbytes)
Path('benchmarks/compute_probe_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
