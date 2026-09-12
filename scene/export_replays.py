"""Export real checkpoint decisions and sampled MaleCNS activity, read-only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib
import numpy as np
import pyarrow.feather as feather
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper
from curriculum import encode,decode
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'scene/dist/data'
out.mkdir(parents=True,exist_ok=True)
run=ROOT/'runs/curriculum-001'
state=json.loads((run/'state.json').read_text())
checkpoint=run/state['checkpoint']
ids=np.load(ROOT/'data/processed/neuron_ids.npy')
a=feather.read_table(ROOT/'data/raw/annotations.feather',columns=['bodyId','somaLocation','tosomaLocation','superclass'])
valid=[]
for row in a.to_pylist():
 k=int(np.searchsorted(ids,row['bodyId']))
 pos=row['somaLocation'] or row['tosomaLocation']
 if k<len(ids) and ids[k]==row['bodyId'] and pos and len(pos)==3:
  valid.append((k,pos,row['superclass']))
rng=np.random.default_rng(778)
chosen=sorted(rng.choice(len(valid),min(3200,len(valid)),replace=False))
selected=[valid[i] for i in chosen]
indices=np.array([x[0] for x in selected])
positions=np.array([x[1] for x in selected],dtype=float)
center=(positions.max(0)+positions.min(0))/2
scale=np.ptp(positions,axis=0).max()/3.5
positions=(positions-center)/scale
# Anatomical coordinates: retain full CNS, display axes separately from anatomy.
positions=positions[:,[0,2,1]]
positions[:,1]*=-1
graph=sparse.load_npz(ROOT/'data/processed/graph.npz')
p=BrainPolicy(graph)
p.load(checkpoint)
base=np.load(run/'initial-expanded.npz')['log_gain']
sub=graph[indices,:][:,indices].tocoo()
sel=np.arange(sub.nnz)
if len(sel)>6500:sel=rng.choice(sel,6500,replace=False)
edges=[[int(sub.col[i]),int(sub.row[i]),float(sub.data[i]),float(np.exp(p.log_gain[indices[sub.col[i]]]-base[indices[sub.col[i]]]))] for i in sel]
brain={'source':'MaleCNS v1.0, original soma locations and retained connections',
       'coordinate_units':'normalized from original EM voxel coordinates',
       'total_neurons':len(ids),'total_edges':int(graph.nnz),'sample_count':len(indices),
       'neurons':[{'id':int(ids[k]),'index':int(k),'position':pos.round(5).tolist(),'group':group} for (k,_,group),pos in zip(selected,positions)],'edges':edges}
(out/'brain.json').write_text(json.dumps(brain,separators=(',',':')))
replays=[]
# Fixed seeds, interleaved sizes. Keep all outcomes; never select for wins.
schedule=[(size,mines,2001000000+group*1000+j) for j in range(6) for group,(size,mines) in enumerate([(5,3),(9,10),(16,40)])]
for size,mines,seed in schedule:
 env=Minesweeper(seed,size,mines)
 acts=np.random.default_rng(seed+10000)
 frames=[]
 while not env.done:
  obs,mask=encode(env)
  probs,cache=p.forward(obs,mask,True)
  chosen_action=int(acts.choice(len(probs),p=probs))
  action=decode(chosen_action,size)
  h=cache[0][-1][indices]
  frame={'visible':env.visible.tolist(),'action':action,'probability':float(probs[chosen_action]),
         'activity':h.astype(float).round(6).tolist(),
         'activity_steps':[x[indices].astype(float).round(6).tolist() for x in cache[0][1:]]}
  env.step(action)
  frame['after']=env.visible.tolist()
  frame['outcome']='win' if env.won else ('mine' if env.done else 'continue')
  frames.append(frame)
 if not frames:continue
 replays.append({'id':f'{size}x{size}-{seed}','size':size,'mines':mines,'seed':seed,'won':bool(env.won),'frames':frames})
 print(size,len(frames),'won',env.won,flush=True)
metadata={'checkpoint':checkpoint.name,'episode':state['episode'],'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'mode':'Recorded evaluation; not live training actions','activity_kind':'signed tanh rate activity; not spikes',
          'selection':'18 fixed seeds interleaved by size; all nonautomatic games retained, no selection for wins',
          'activity_scale':0.1,'replays':replays}
temp=out/'replays.tmp.json'
temp.write_text(json.dumps(metadata,separators=(',',':')))
temp.replace(out/'replays.json')
(out/'status-snapshot.json').write_text((run/'progress.json').read_text())
print('Exported',len(indices),'neurons',len(edges),'edges',checkpoint.name,flush=True)
