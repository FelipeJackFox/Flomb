"""Extend three spatial heads to at most 5000 updates; preserve Adam by replay.

Old heads did not save optimizer state. Reconstruct first 1000 updates using the
same seed/minibatches and demand bit-exact parameter equality before continuing.
New game benchmark is reserved before any training, excluded from model selection.
"""
import hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy,SEEDS
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import write_json
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
from experiments.scaled_data import identity
from minesweeper import Minesweeper
OUT=Path('runs/spatial-extended-001');BASE=Path('runs/structural-learning-001');OLD=Path('runs/spatial-consistency-001');CACHE=Path('runs/spatial-decoder-001')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reserve():
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/games.json'):
  if p.parent!=OUT:used.update(r['layout_hash'] for r in json.loads(p.read_text()))
 games=[];seed=5600700000
 while len(games)<500:
  env=Minesweeper(seed,7,7);key=identity(env)
  if key not in used:games.append(dict(seed=seed,size=7,mines=7,layout_hash=key));used.add(key)
  seed+=1
 return games

@torch.no_grad()
def validate(head,cache):
 a,l,c,y=cache
 return sum(float(equivalent_loss(head(a[i:i+128],c[i:i+128]),l[i:i+128],y[i:i+128]))*len(a[i:i+128]) for i in range(0,len(a),128))/len(a)

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise SystemExit('Preserve existing run')
 protected=[Path('runs/hybrid-001/checkpoint.pkl'),BASE/'training/retina_plastic-20260926.pt',CACHE/'train.pt',CACHE/'holdout.pt',*[OLD/f'spatial-{s}.pt' for s in SEEDS]]
 hashes={str(p):digest(p) for p in protected}
 manifest=dict(seeds=SEEDS,max_updates=5000,base_updates=1000,batch=64,lr=.001,selection='min validation loss; checks every250 after1000',early_stop='6 checks without >=1e-4 validation improvement',optimizer='Adam reconstructed with exact first1000-update replay',hashes=hashes,torch=torch.__version__)
 write_json(OUT/'manifest.json',manifest);source=OUT/'source';source.mkdir()
 for p in [Path(__file__),Path('experiments/spatial_decoder.py'),Path('experiments/validate_spatial_decoder.py'),Path('experiments/scaled_train.py')]:shutil.copy2(p,source/p.name)
 games=reserve();write_json(OUT/'games.json',games)
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());cache={s:torch.load(CACHE/f'{s}.pt',weights_only=False) for s in ('train','holdout')}
 groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)];selection=[];heads=[]
 for seed in SEEDS:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);head=ActivityHead(True);optim=torch.optim.Adam(head.parameters(),lr=.001)
  old=torch.load(OLD/f'spatial-{seed}.pt',weights_only=False);assert old['step']==1000
  baseline=ActivityHead(True);baseline.load_state_dict(old['head']);heads.append((f'baseline-{seed}',baseline.eval()))
  history=[];best=float('inf');patience=0;anchor=float('inf');started=time.monotonic()
  for step in range(1,5001):
   ix=[int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(64)];a,l,c,y=[v[ix] for v in cache['train']]
   optim.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optim.step()
   if step==1000:
    for k,v in head.state_dict().items():torch.testing.assert_close(v,old['head'][k],rtol=0,atol=0)
   if step>=1000 and step%250==0:
    value=validate(head,cache['holdout'])
    if value<best:
     best=value;chosen=step;weights={k:v.detach().clone() for k,v in head.state_dict().items()}
     torch.save(dict(head=weights,optimizer=optim.state_dict(),rng=rng.bit_generator.state,step=step,seed=seed,validation_loss=value),OUT/f'best-{seed}.pt')
    if value<anchor-1e-4:anchor=value;patience=0
    else:patience+=1
    record=dict(phase='head_training',seed=seed,step=step,validation_loss=value,best_step=chosen,patience=patience,seconds=time.monotonic()-started);history.append(record);write_json(OUT/'progress.json',record);print(record,flush=True)
    tmp=OUT/f'latest-{seed}.tmp';torch.save(dict(head=head.state_dict(),optimizer=optim.state_dict(),rng=rng.bit_generator.state,step=step,seed=seed,history=history),tmp);tmp.replace(OUT/f'latest-{seed}.pt')
    if patience>=6:break
  head.load_state_dict(weights);heads.append((f'extended-{seed}',head.eval()));write_json(OUT/f'history-{seed}.json',history)
  selection.append(dict(seed=seed,chosen_step=chosen,stopped_step=step,validation_loss=best,baseline_validation_loss=old['validation_loss'],first1000_replay_exact=True,seconds=time.monotonic()-started))
 write_json(OUT/'selection.json',selection);del cache
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);mapping=pickle.loads((BASE/'mapping.pkl').read_bytes());brain=RetinaPolicy(op,mapping,True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 results={}
 for name,head in heads:
  write_json(OUT/'progress.json',dict(phase='autonomous_evaluation',variant=name,games=500));start=time.monotonic()
  rows=evaluate_games(MemoPolicy(memo,head),games);write_json(OUT/f'{name}-games.json',rows);play=[r for r in rows if not r['automatic']]
  results[name]=dict(wins=sum(r['won'] for r in play),n=len(play),seconds=time.monotonic()-start);write_json(OUT/'results.json',results);print(name,results[name],flush=True)
 for p,d in hashes.items():assert digest(p)==d
 write_json(OUT/'verification.json',dict(original_hashes_unchanged=True,all_three_optimizer_replays_exact=True,memo_hits=memo.hits,memo_misses=memo.misses))
 write_json(OUT/'completed.json',dict(completed=True));op.close()
if __name__=='__main__':main()
