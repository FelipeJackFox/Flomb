"""Three beta=0 DAgger rounds, frozen brain, old-data update-matched control."""
import copy,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.extend_spatial_decoder import digest,validate,BASE,CACHE
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import arrays,write_json
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
from experiments.scaled_data import identity
from experiments.train import visible,teacher,public_context
from experiments.backbone import encode_visible
from solver import analyze
from minesweeper import Minesweeper
OUT=Path('runs/spatial-dagger-001');SEED=20261002

def reserve():
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/games.json'):
  used.update(r['layout_hash'] for r in json.loads(p.read_text()))
 prior=set(used);result={};seed=5800700000
 for split,n in [('games',500),('collection',900)]:
  rows=[]
  while len(rows)<n:
   env=Minesweeper(seed,7,7);key=identity(env)
   if key not in used:rows.append(dict(seed=seed,size=7,mines=7,layout_hash=key));used.add(key)
   seed+=1
  result[split]=rows
 assert not ({r['layout_hash'] for r in result['games']} & {r['layout_hash'] for r in result['collection']})
 assert not prior & {r['layout_hash'] for rows in result.values() for r in rows}
 return result

@torch.no_grad()
def collect(memo,head,games):
 rows=[];maps=[];stats=dict(games=len(games),wins=0,clicks=0,safe_opportunities=0,safe_choices=0,teacher_actions=0)
 for start in range(0,len(games),16):
  active=[(Minesweeper(g['seed'],7,7),g) for g in games[start:start+16]]
  while active:
   active=[(e,g) for e,g in active if not e.done]
   if not active:break
   x,legal=encode_visible([visible(e) for e,g in active]);c=torch.tensor(np.array([public_context(7,7) for _ in active]));a=memo.get(torch.from_numpy(x),c)
   actions=head(a,c).masked_fill(~torch.from_numpy(legal),-1e9).argmax(1).tolist()
   for i,((env,g),action) in enumerate(zip(active,actions)):
    info=analyze(env.observation(),env.legal_mask(),7,7)
    native=(action//16)*7+action%16
    if info.safe:
     labels,_=teacher(env,info)
     assert labels.any() and np.all(~labels | legal[i])
     rows.append(dict(**g,board=visible(env).copy(),labels=labels,learner_action=action,learner_safe=bool(native in info.safe)))
     maps.append(a[i].clone());stats['safe_opportunities']+=1;stats['safe_choices']+=int(native in info.safe)
    env.step(native);stats['clicks']+=1
    if env.done:stats['wins']+=int(env.won)
  print('collection',start+len(games[start:start+16]),stats,flush=True)
 assert rows
 _,l,c,y=arrays(rows)
 return rows,(torch.stack(maps),l,c,y),stats

def main(out=OUT,seed=SEED,splits=None):
 OUT=Path(out);SEED=seed
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise SystemExit('Preserve existing run')
 paths=[Path('runs/spatial-extended-001')/f'best-{SEED}.pt',BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',CACHE/'train.pt',CACHE/'holdout.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 hashes={str(p):digest(p) for p in paths}
 write_json(OUT/'manifest.json',dict(seed=SEED,rounds=3,games_per_round=300,updates_per_round=750,batch=64,new_fraction=.5,teacher_actions=0,labels='certified safe only',selection='old holdout minimum loss including baseline',hashes=hashes))
 (OUT/'source').mkdir()
 for p in [Path(__file__),Path('experiments/validate_spatial_decoder.py'),Path('experiments/scaled_train.py'),Path('solver.py'),Path('minesweeper.py')]:shutil.copy2(p,OUT/'source'/p.name)
 splits=reserve() if splits is None else splits;write_json(OUT/'games.json',splits['games']);write_json(OUT/'collection-games.json',splits['collection']);write_json(OUT/'split_verification.json',dict(unique=len({r['layout_hash'] for rows in splits.values() for r in rows}),train_test_overlap=len({r['layout_hash'] for r in splits['games']} & {r['layout_hash'] for r in splits['collection']}),collection_provenance='See parent protocol when supplied externally'))
 old=torch.load(paths[0],weights_only=False);cache=torch.load(CACHE/'train.pt',weights_only=False);valid=torch.load(CACHE/'holdout.pt',weights_only=False)
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)]
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(paths[1],weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 heads={};opts={};rngs={};best={};selected={};history=[];allrows=[];newcache=None
 for name in ('baseline','control','dagger'):
  h=ActivityHead(True);h.load_state_dict(old['head']);heads[name]=h
  if name!='baseline':
   opts[name]=torch.optim.Adam(h.parameters(),lr=.001);opts[name].load_state_dict(copy.deepcopy(old['optimizer']));rngs[name]=np.random.default_rng(SEED+77)
   best[name]=validate(h,valid);selected[name]=dict(head=copy.deepcopy(h.state_dict()),optimizer=copy.deepcopy(opts[name].state_dict()),rng=rngs[name].bit_generator.state,step=0,validation_loss=best[name])
 for round_ in range(1,4):
  write_json(OUT/'progress.json',dict(phase='collection',round=round_))
  rows,fresh,stats=collect(memo,heads['dagger'],splits['collection'][(round_-1)*300:round_*300]);allrows.extend(rows)
  newcache=fresh if newcache is None else tuple(torch.cat([a,b]) for a,b in zip(newcache,fresh))
  with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(train=allrows),f)
  write_json(OUT/f'collection-{round_}.json',stats)
  for name in ('control','dagger'):
   h=heads[name];opt=opts[name];rng=rngs[name]
   for local in range(1,751):
    step=(round_-1)*750+local;nold=64 if name=='control' else 32
    ix=[int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(nold)];batch=[v[ix] for v in cache]
    if name=='dagger':
     ix=rng.integers(len(newcache[0]),size=32);batch=[torch.cat([v,w[ix]]) for v,w in zip(batch,newcache)]
    a,l,c,y=batch;opt.zero_grad();loss=equivalent_loss(h(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True);opt.step()
    if local%250==0:
     val=validate(h,valid);record=dict(phase='training',variant=name,round=round_,step=step,validation_loss=val,aggregated_positions=len(allrows));history.append(record);write_json(OUT/'progress.json',record);print(record,flush=True)
     state=dict(head=copy.deepcopy(h.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,validation_loss=val)
     if val<best[name]:best[name]=val;selected[name]=state
     torch.save(state,OUT/f'latest-{name}.pt')
   torch.save(selected[name],OUT/f'best-{name}.pt')
  write_json(OUT/'history.json',history)
 results={}
 for name,h in heads.items():
  if name!='baseline':h.load_state_dict(selected[name]['head'])
  write_json(OUT/'progress.json',dict(phase='evaluation',variant=name));rows=evaluate_games(MemoPolicy(memo,h),splits['games']);write_json(OUT/f'{name}-games.json',rows)
  results[name]=dict(wins=sum(r['won'] for r in rows),n=len(rows),selected_step=0 if name=='baseline' else selected[name]['step']);write_json(OUT/'results.json',results);print(name,results[name],flush=True)
 assert all(digest(p)==sha for p,sha in hashes.items())
 write_json(OUT/'verification.json',dict(original_hashes_unchanged=True,teacher_actions=0,memo_hits=memo.hits,memo_misses=memo.misses))
 write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
