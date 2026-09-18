"""More DAgger experience vs equal updates on a frozen existing data buffer."""
import copy,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.spatial_dagger import collect
from experiments.extend_spatial_decoder import BASE,CACHE,digest,validate
from experiments.validate_spatial_decoder import SEEDS,ActivityMemo,MemoPolicy
from experiments.spatial_decoder import ActivityHead
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import arrays,write_json
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
from experiments.scaled_data import identity
from minesweeper import Minesweeper
OUT=Path('runs/spatial-dagger-expanded-001')
def origin(seed):return Path('runs/spatial-dagger-001' if seed==SEEDS[0] else f'runs/spatial-dagger-{seed}')
def target(seed):return Path(f'runs/spatial-dagger-expanded-{seed}')
def reserve():
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for pattern in ('*/games.json','*/collection-games.json'):
  for p in Path('runs').glob(pattern):used.update(r['layout_hash'] for r in json.loads(p.read_text()))
 result={};seed=6200700000
 for name,n in [('games',500),('collection',1800)]:
  rows=[]
  while len(rows)<n:
   key=identity(Minesweeper(seed,7,7))
   if key not in used:rows.append(dict(seed=seed,size=7,mines=7,layout_hash=key));used.add(key)
   seed+=1
  result[name]=rows
 return result

def checkpoint(head,opt,rng,step,loss,**extra):
 return dict(head=copy.deepcopy(head.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,validation_loss=loss,**extra)
def save(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
@torch.no_grad()
def features(memo,rows):
 chunks=[]
 for i in range(0,len(rows),16):
  x,l,c,y=arrays(rows[i:i+16]);chunks.append((memo.get(x,c),l,c,y))
 return tuple(torch.cat([part[j] for part in chunks]) for j in range(4))
def train_seed(seed,memo,splits,cache,valid,groups):
 out=target(seed);out.mkdir(exist_ok=False);old=torch.load(origin(seed)/'best-dagger.pt',weights_only=False)
 inherited=pickle.loads((origin(seed)/'dataset.pkl').read_bytes())['train']
 write_json(OUT/'progress.json',dict(phase='cache_existing_experience',seed=seed,positions=len(inherited)))
 fixed=features(memo,inherited);torch.save(fixed,out/'inherited.pt');aggregate=fixed;rows=list(inherited)
 write_json(out/'manifest.json',dict(seed=seed,parent=str(origin(seed)),parent_checkpoint_sha256=digest(origin(seed)/'best-dagger.pt'),parent_step=old['step'],rounds=3,games_per_round=600,updates_per_round=750,batch=64,original_fraction=.5,control='32 original +32 fixed inherited DAgger',expanded='32 original +32 aggregate inherited and new DAgger',teacher_actions=0))
 write_json(out/'games.json',splits['games']);write_json(out/'collection-games.json',splits['collection'])
 heads={};opts={};rngs={};selected={};history=[]
 for name in ('baseline','control','expanded'):
  h=ActivityHead(True);h.load_state_dict(old['head']);heads[name]=h
  if name!='baseline':
   opt=torch.optim.Adam(h.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(old['optimizer']));opts[name]=opt
   rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(old['rng']);rngs[name]=rng
   selected[name]=checkpoint(h,opt,rng,0,validate(h,valid),parent_step=old['step'])
   save(out/f'best-{name}.pt',selected[name])
 for rnd in range(1,4):
  progress=dict(phase='collection',seed=seed,round=rnd,games=600);write_json(OUT/'progress.json',progress);write_json(out/'progress.json',progress)
  freshrows,fresh,stats=collect(memo,heads['expanded'],splits['collection'][(rnd-1)*600:rnd*600]);rows.extend(freshrows)
  aggregate=tuple(torch.cat([a,b]) for a,b in zip(aggregate,fresh))
  torch.save(fresh,out/f'round-{rnd}-activity.pt')
  with (out/f'round-{rnd}-positions.pkl').open('wb') as f:pickle.dump(freshrows,f)
  with (out/'dataset.pkl').open('wb') as f:pickle.dump(dict(train=rows),f)
  write_json(out/f'collection-{rnd}.json',stats)
  for name in ('control','expanded'):
   head=heads[name];opt=opts[name];rng=rngs[name];experience=fixed if name=='control' else aggregate
   for local in range(1,751):
    step=(rnd-1)*750+local
    ix=[int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(32)];jx=rng.integers(len(experience[0]),size=32)
    a,l,c,y=[torch.cat([v[ix],w[jx]]) for v,w in zip(cache,experience)]
    opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
    if local%250==0:
     val=validate(head,valid);state=checkpoint(head,opt,rng,step,val,parent_step=old['step']);save(out/f'latest-{name}.pt',state)
     if val<selected[name]['validation_loss']:selected[name]=state;save(out/f'best-{name}.pt',state)
     progress=dict(phase='training',seed=seed,variant=name,round=rnd,step=step,validation_loss=val,selected_step=selected[name]['step'],positions=len(rows),new_positions=len(rows)-len(inherited));history.append(progress)
     write_json(OUT/'progress.json',progress);write_json(out/'progress.json',progress);print(progress,flush=True)
  write_json(out/'history.json',history)
 result={}
 for name,head in heads.items():
  if name!='baseline':head.load_state_dict(selected[name]['head'])
  progress=dict(phase='evaluation',seed=seed,variant=name);write_json(OUT/'progress.json',progress);write_json(out/'progress.json',progress)
  played=evaluate_games(MemoPolicy(memo,head),splits['games']);write_json(out/f'{name}-games.json',played)
  result[name]=dict(wins=sum(r['won'] for r in played),n=len(played),selected_step=0 if name=='baseline' else selected[name]['step']);write_json(out/'results.json',result);print(seed,name,result[name],flush=True)
 write_json(out/'completed.json',dict(completed=True));write_json(out/'progress.json',dict(phase='completed'))
 return result

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists() or any(target(s).exists() for s in SEEDS):raise SystemExit('Preserve existing run')
 paths=[BASE/'training/retina_plastic-20260926.pt',BASE/'mapping.pkl',BASE/'dataset.pkl',CACHE/'train.pt',CACHE/'holdout.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 for s in SEEDS:paths.extend([origin(s)/'best-dagger.pt',origin(s)/'dataset.pkl'])
 hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(seeds=SEEDS,hashes=hashes,rounds=3,games_per_round=600,updates_per_round=750,selection='minimum original holdout loss every250 including baseline',torch=torch.__version__))
 src=OUT/'source';src.mkdir()
 for p in [Path(__file__),*[Path('experiments')/n for n in ('spatial_dagger.py','spatial_decoder.py','retina_policy.py','plastic_sparse.py','validate_spatial_decoder.py','scaled_train.py','extend_spatial_decoder.py','capacity_probe.py','expressive_models.py','scaled_data.py','train.py','backbone.py')],Path('solver.py'),Path('minesweeper.py')]:shutil.copy2(p,src/p.name)
 splits=reserve();write_json(OUT/'games.json',splits['games']);write_json(OUT/'collection-games.json',splits['collection']);write_json(OUT/'split_verification.json',dict(final_unique=500,collection_unique=1800,prior_overlap=0,train_test_overlap=0))
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)]
 cache=torch.load(CACHE/'train.pt',weights_only=False);valid=torch.load(CACHE/'holdout.pt',weights_only=False)
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 results={}
 for seed in SEEDS:
  results[str(seed)]=train_seed(seed,memo,splits,cache,valid,groups);write_json(OUT/'results.json',results)
 assert all(digest(p)==sha for p,sha in hashes.items())
 write_json(OUT/'verification.json',dict(original_hashes_unchanged=True,memo_hits=memo.hits,memo_misses=memo.misses,teacher_actions=0))
 write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
