"""Paired head-only pilot. Cache frozen neural maps; fresh layout-disjoint games."""
import hashlib,json,pickle,time,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.spatial_decoder import activity_map,ActivityHead,FrozenPolicy
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import arrays,write_json
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
from experiments.scaled_data import identity
from minesweeper import Minesweeper

OUT=Path('runs/spatial-decoder-001');ROOT=Path('runs/structural-learning-001')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'completed.json').exists():raise SystemExit('Completed run exists; preserve outputs.')
 protected='runs/hybrid-001/checkpoint.pkl';checkpoint=ROOT/'training/retina_plastic-20260926.pt'
 data=pickle.loads((ROOT/'dataset.pkl').read_bytes());mapping=pickle.loads((ROOT/'mapping.pkl').read_bytes())
 manifest=dict(seed=20261002,updates=1000,batch=64,lr=.001,selection='minimum validation loss every 100 updates',checkpoint_sha256=digest(checkpoint),protected_sha256=digest(protected),dataset_sha256=digest(ROOT/'dataset.pkl'),frozen='encoder, node/edge gains, graph, neuronal mapping',input='neural activity + board size/mine count only',torch=torch.__version__)
 if (OUT/'manifest.json').exists():
  assert json.loads((OUT/'manifest.json').read_text())==manifest
 else:write_json(OUT/'manifest.json',manifest)
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
 brain=RetinaPolicy(op,mapping,True);brain.load_state_dict(torch.load(checkpoint,weights_only=False)['model']);brain.requires_grad_(False);brain.eval()
 cache={}
 for split in ('train','holdout'):
  path=OUT/f'{split}.pt'
  if path.exists():
   cache[split]=torch.load(path,weights_only=False);continue
  maps=[];legals=[];contexts=[];labels=[]
  for i in range(0,len(data[split]),16):
   x,l,c,y=arrays(data[split][i:i+16]);a=activity_map(brain,x,c);maps.append(a);legals.append(l);contexts.append(c);labels.append(y)
   if i==0:
    with torch.no_grad():
     logits=brain(x,c)
     restored=brain.decoder(torch.cat([a.permute(0,2,3,1),c[:,None,None].expand(-1,16,16,-1)],-1)).squeeze(-1).flatten(1)
     torch.testing.assert_close(restored[l],logits[l],rtol=1e-5,atol=1e-5)
   if i%1024==0:
    p=dict(phase='cache',split=split,done=i,target=len(data[split]));write_json(OUT/'progress.json',p);print(p,flush=True)
  cache[split]=tuple(torch.cat(v) for v in (maps,legals,contexts,labels));torch.save(cache[split],path)
 # Reserve novel layouts against all previous saved evaluation and training lists.
 used=set()
 for path in Path('runs').glob('*/dataset.pkl'):
  old=pickle.loads(path.read_bytes())
  for rows in old.values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 games=[]
 for size,mines in [(5,3),(7,7)]:
  seed=5200000000+size*100000;count=0;attempts=0
  import math
  target=min(250,math.comb(size*size-1,mines)-sum(k.startswith(f'{size}:') for k in used))
  if target<=0:raise ValueError('No unseen layouts remain')
  while count<target:
   attempts+=1
   if attempts>100000:raise RuntimeError('Fresh layout search exhausted')
   env=Minesweeper(seed,size,mines);key=identity(env)
   if key not in used:games.append(dict(seed=seed,size=size,mines=mines,layout_hash=key));used.add(key);count+=1
   seed+=1
 write_json(OUT/'games.json',games)
 heads={};results=[]
 groups={}
 for i,row in enumerate(data['train']):groups.setdefault((row['size'],row['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)]
 for name,spatial in [('pointwise',False),('spatial',True)]:
  torch.manual_seed(manifest['seed']);rng=np.random.default_rng(manifest['seed']);head=ActivityHead(spatial)
  optimizer=torch.optim.Adam(head.parameters(),lr=.001);best=float('inf');history=[];started=time.monotonic()
  for step in range(1,1001):
   ids=[int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(64)]
   a,l,c,y=[v[ids] for v in cache['train']]
   optimizer.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
   if step%100==0:
    with torch.no_grad():
     a,l,c,y=cache['holdout'];vl=[]
     for i in range(0,len(a),128):vl.append(float(equivalent_loss(head(a[i:i+128],c[i:i+128]),l[i:i+128],y[i:i+128]))*len(a[i:i+128]))
     value=sum(vl)/len(a)
    if value<best:
     best=value;chosen=step;weights={k:v.detach().clone() for k,v in head.state_dict().items()}
     torch.save(dict(head=weights,step=step,validation_loss=value,manifest=manifest),OUT/f'{name}.pt')
    p=dict(phase='training_head',variant=name,step=step,loss=float(loss.detach()),validation_loss=value,seconds=time.monotonic()-started);history.append(p);write_json(OUT/'progress.json',p);print(p,flush=True)
  head.load_state_dict(weights);head.eval();heads[name]=head
  write_json(OUT/f'{name}-history.json',history)
  results.append(dict(variant=name,selected_step=chosen,validation_loss=best,parameters=sum(p.numel() for p in head.parameters())))
 # Frozen backbone hashes certify the caching/training path did not mutate it.
 assert digest(checkpoint)==manifest['checkpoint_sha256'] and digest(protected)==manifest['protected_sha256']
 for name,model in [('original',brain),*[(n,FrozenPolicy(brain,h)) for n,h in heads.items()]]:
  write_json(OUT/'progress.json',dict(phase='autonomous_evaluation',variant=name,games=len(games)))
  rows=evaluate_games(model,games)
  write_json(OUT/f'{name}-games.json',rows);print(name,'evaluation done',flush=True)
 write_json(OUT/'results.json',results);write_json(OUT/'completed.json',dict(completed=True,protected_sha256=digest(protected)))
 source=OUT/'source';source.mkdir()
 for p in [Path(__file__),Path('experiments/spatial_decoder.py'),Path('experiments/retina_policy.py'),Path('experiments/scaled_train.py')]:shutil.copy2(p,source/p.name)
 op.close()
if __name__=='__main__':main()
