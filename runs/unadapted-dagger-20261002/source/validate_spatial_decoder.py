"""Three head seeds, one frozen backbone, fresh shared 7x7 game benchmark.

Two new seeds retain the pilot's exact budget. The first seed is reused as-is.
Neural state memoization is inference-only, keyed by full observation/context.
"""
import hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import sparse
from experiments.spatial_decoder import activity_map,ActivityHead
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import write_json
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
from experiments.scaled_data import identity
from minesweeper import Minesweeper

OUT=Path('runs/spatial-consistency-001');PILOT=Path('runs/spatial-decoder-001');BASE=Path('runs/structural-learning-001')
SEEDS=(20261002,20261003,20261004)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class ActivityMemo:
 def __init__(self,brain):self.brain=brain;self.cache={};self.hits=0;self.misses=0
 @torch.no_grad()
 def get(self,x,context):
  keys=[hashlib.sha256(a.numpy().tobytes()+b.numpy().tobytes()).digest() for a,b in zip(x,context)]
  missing=list(dict.fromkeys(k for k in keys if k not in self.cache))
  if missing:
   ix=[keys.index(k) for k in missing]
   maps=activity_map(self.brain,x[ix],context[ix])
   for key,a in zip(missing,maps):self.cache[key]=a.clone()
  self.misses+=len(missing);self.hits+=len(keys)-len(missing)
  return torch.stack([self.cache[k] for k in keys])

class OriginalHead(nn.Module):
 def __init__(self,brain):super().__init__();self.decoder=brain.decoder
 def forward(self,a,c):
  return self.decoder(torch.cat([a.permute(0,2,3,1),c[:,None,None].expand(-1,16,16,-1)],-1)).squeeze(-1).flatten(1)

class MemoPolicy(nn.Module):
 def __init__(self,memo,head):super().__init__();self.memo=memo;self.head=head
 def forward(self,x,c):return self.head(self.memo.get(x,c),c)


def reserve_games():
 used=set()
 for path in Path('runs').glob('*/dataset.pkl'):
  d=pickle.loads(path.read_bytes())
  for rows in d.values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for path in Path('runs').glob('*/games.json'):
  if path.parent==OUT:continue
  rows=json.loads(path.read_text())
  if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if 'layout_hash' in r)
 games=[];seed=5400700000
 for _ in range(100000):
  env=Minesweeper(seed,7,7);key=identity(env)
  if key not in used:games.append(dict(size=7,mines=7,seed=seed,layout_hash=key));used.add(key)
  seed+=1
  if len(games)==500:break
 assert len(games)==500
 return games


def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise SystemExit('Preserve existing run')
 paths=[PILOT/'train.pt',PILOT/'holdout.pt',BASE/'dataset.pkl',BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 hashes={str(p):digest(p) for p in paths}
 write_json(OUT/'manifest.json',dict(seeds=SEEDS,new_seeds=SEEDS[1:],updates=1000,batch=64,lr=.001,selection='minimum validation loss every100',input_hashes=hashes,torch=torch.__version__,scope='head seed consistency, not independently retrained brains'))
 source=OUT/'source';source.mkdir()
 for p in [Path(__file__),Path('experiments/spatial_decoder.py'),Path('experiments/retina_policy.py'),Path('experiments/plastic_sparse.py'),Path('experiments/scaled_train.py'),Path('minesweeper.py'),Path('solver.py')]:shutil.copy2(p,source/p.name)
 games=reserve_games();write_json(OUT/'games.json',games)
 cache={s:torch.load(PILOT/f'{s}.pt',weights_only=False) for s in ('train','holdout')}
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)];heads=[];selection=[]
 for seed in SEEDS:
  for name,spatial in [('pointwise',False),('spatial',True)]:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);head=ActivityHead(spatial)
   if seed==SEEDS[0]:
    saved=torch.load(PILOT/f'{name}.pt',weights_only=False);head.load_state_dict(saved['head']);best=saved['validation_loss'];chosen=saved['step'];seconds=0.
   else:
    optimizer=torch.optim.Adam(head.parameters(),lr=.001);best=float('inf');history=[];start=time.monotonic()
    for step in range(1,1001):
     ix=[int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(64)];a,l,c,y=[v[ix] for v in cache['train']]
     optimizer.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
     if step%100==0:
      with torch.no_grad():
       a,l,c,y=cache['holdout'];value=sum(float(equivalent_loss(head(a[i:i+128],c[i:i+128]),l[i:i+128],y[i:i+128]))*len(a[i:i+128]) for i in range(0,len(a),128))/len(a)
      if value<best:best=value;chosen=step;weights={k:v.detach().clone() for k,v in head.state_dict().items()}
      record=dict(phase='head_training',variant=name,seed=seed,step=step,validation_loss=value,seconds=time.monotonic()-start);history.append(record);write_json(OUT/'progress.json',record);print(record,flush=True)
    seconds=time.monotonic()-start;head.load_state_dict(weights);write_json(OUT/f'{name}-{seed}-history.json',history)
   torch.save(dict(head=head.state_dict(),seed=seed,step=chosen,validation_loss=best),OUT/f'{name}-{seed}.pt')
   heads.append((f'{name}-{seed}',head.eval()));selection.append(dict(variant=name,seed=seed,step=chosen,validation_loss=best,seconds=seconds))
 write_json(OUT/'selection.json',selection)
 del cache
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);mapping=pickle.loads((BASE/'mapping.pkl').read_bytes())
 brain=RetinaPolicy(op,mapping,True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval()
 memo=ActivityMemo(brain)
 # Verify cache and original decoder equivalence with mixed 5x5/7x7 observations.
 from experiments.capacity_probe import arrays
 rows=[next(r for r in data['holdout'] if r['size']==s) for s in (5,7)]
 x,l,c,y=arrays(rows)
 with torch.no_grad():
  expected=brain(x,c);a=memo.get(x,c);actual=OriginalHead(brain)(a,c)
  torch.testing.assert_close(actual[l],expected[l],rtol=1e-5,atol=1e-5)
  torch.testing.assert_close(memo.get(x.flip(0),c.flip(0)),a.flip(0),rtol=0,atol=0)
 results={}
 for name,head in [('original',OriginalHead(brain)),*heads]:
  write_json(OUT/'progress.json',dict(phase='autonomous_evaluation',variant=name,games=len(games)));start=time.monotonic()
  rows=evaluate_games(MemoPolicy(memo,head),games);write_json(OUT/f'{name}-games.json',rows)
  play=[r for r in rows if not r['automatic']];record=dict(wins=sum(r['won'] for r in play),n=len(play),seconds=time.monotonic()-start,memo_hits=memo.hits,memo_misses=memo.misses)
  results[name]=record;write_json(OUT/'results.json',results);print(name,record,flush=True)
 for path,d in hashes.items():assert digest(path)==d
 write_json(OUT/'verification.json',dict(protected_hashes_unchanged=True,original_decoder_parity=True,memo_reorder_exact=True,memo_hits=memo.hits,memo_misses=memo.misses))
 write_json(OUT/'completed.json',dict(completed=True));op.close()
if __name__=='__main__':main()
