"""Bounded batch-size throughput sweep, full backbone forward/backward."""
import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from experiments.backbone import Backbone,encode_visible
from experiments.train import visible
from minesweeper import Minesweeper
b=Backbone(sparse.load_npz('data/processed/graph.npz'),'runs/curriculum-001/initial-expanded.npz');results=[]
for batch in [16,32,64]:
 x,_=encode_visible([visible(Minesweeper(77+i,16,40)) for i in range(batch)]);df=np.ones((batch,len(b.output_index)),np.float32)/batch
 t=time.perf_counter();features,cache=b.forward(x);dg=b.backward(cache,df);elapsed=time.perf_counter()-t
 assert np.isfinite(features).all() and np.isfinite(dg).all()
 results.append({'batch':batch,'seconds':elapsed,'seconds_per_sample':elapsed/batch,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 del cache,features,dg
Path('benchmarks/batch_sizes_result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
