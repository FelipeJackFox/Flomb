"""Full forward+backward batched versus serial, same graph, states and gradients."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from experiments.backbone import Backbone,encode_visible
from experiments.train import visible
from minesweeper import Minesweeper
b=Backbone(sparse.load_npz('data/processed/graph.npz'),'runs/curriculum-001/initial-expanded.npz')
x,_=encode_visible([visible(Minesweeper(100+i,16,40)) for i in range(16)])
u=np.random.default_rng(88).normal(size=(16,len(b.output_index))).astype(np.float32)/16
# Warmup, then compare numerical work without changing parameters.
b.forward(x,False)
t=time.perf_counter();f,c=b.forward(x);g=b.backward(c,u);batched=time.perf_counter()-t
fs=[];gs=[];t=time.perf_counter()
for i in range(16):
 fi,ci=b.forward(x[i:i+1]);fs.append(fi);gs.append(b.backward(ci,u[i:i+1]))
serial=time.perf_counter()-t
np.testing.assert_allclose(f,np.concatenate(fs),rtol=2e-5,atol=2e-6)
np.testing.assert_allclose(g,np.sum(gs,axis=0),rtol=1e-4,atol=2e-6)
r={'batch':16,'forward_backward_batch_seconds':batched,'forward_backward_serial_seconds':serial,'speedup':serial/batched,'max_feature_difference':float(np.max(abs(f-np.concatenate(fs)))),'max_gradient_difference':float(np.max(abs(g-np.sum(gs,axis=0)))),'scope':'Full backbone forward/backward only; same trainable gains, no frozen cache. Shared host with other training. Not overall trainer speedup.'}
Path('benchmarks/backbone_batch_probe_result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
