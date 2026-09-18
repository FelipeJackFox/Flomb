"""Fixed-data comparison: uniform replay vs current known-mine mistakes."""
import copy,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.expand_dagger_experience import target as parent,origin,checkpoint,save,reserve,BASE,CACHE,digest,SEEDS
from experiments.diagnose_mine_choices import info_from
from experiments.spatial_decoder import ActivityHead
from experiments.extend_spatial_decoder import validate
from experiments.expressive_models import equivalent_loss
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.scaled_train import evaluate_games
from experiments.capacity_probe import write_json
OUT=Path('runs/prioritized-mines-001')
def target(seed):return Path(f'runs/prioritized-mines-{seed}')
@torch.no_grad()
def mistakes(head,cache,mines):
 indices=[];safe=0
 for i in range(0,len(cache[0]),128):
  a,l,c,y=[v[i:i+128] for v in cache];act=head(a,c).masked_fill(~l,-1e9).argmax(1);ids=torch.arange(len(act));safe+=int(y[ids,act].sum());indices.extend((torch.nonzero(mines[i:i+len(act)][ids,act]).flatten()+i).tolist())
 return np.asarray(indices,dtype=np.int64),safe

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists() or any(target(s).exists() for s in SEEDS):raise SystemExit('Preserve existing run')
 paths=[BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',CACHE/'train.pt',CACHE/'holdout.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 for s in SEEDS:
  paths += [parent(s)/'best-expanded.pt',parent(s)/'inherited.pt',origin(s)/'dataset.pkl']
  saved=torch.load(parent(s)/'best-expanded.pt',weights_only=False)
  for rnd in range(1,(saved['step']+749)//750+1):paths += [parent(s)/f'round-{rnd}-activity.pt',parent(s)/f'round-{rnd}-positions.pkl']
 hashes={str(p):digest(p) for p in paths}
 write_json(OUT/'manifest.json',dict(seeds=SEEDS,updates=1500,batch=64,lr=.001,selection='original holdout minimum every250 including initial',priority='32 original +16 uniform experience +16 current known-mine errors; refresh every250',control='32 original +32 uniform experience',importance_correction=False,hashes=hashes))
 src=OUT/'source';src.mkdir()
 for p in [Path(__file__),Path('experiments/expand_dagger_experience.py'),Path('experiments/diagnose_mine_choices.py'),Path('experiments/spatial_decoder.py'),Path('experiments/expressive_models.py'),Path('solver.py')]:shutil.copy2(p,src/p.name)
 games=reserve()['games'];write_json(OUT/'games.json',games)
 oldcache=torch.load(CACHE/'train.pt',weights_only=False);valid=torch.load(CACHE/'holdout.pt',weights_only=False);data=pickle.loads((BASE/'dataset.pkl').read_bytes());groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)]
 for seed in SEEDS:
  out=target(seed);out.mkdir();saved=torch.load(parent(seed)/'best-expanded.pt',weights_only=False)
  rows=pickle.loads((origin(seed)/'dataset.pkl').read_bytes())['train'];cache=torch.load(parent(seed)/'inherited.pt',weights_only=False)
  for rnd in range(1,(saved['step']+749)//750+1):
   rows.extend(pickle.loads((parent(seed)/f'round-{rnd}-positions.pkl').read_bytes()));part=torch.load(parent(seed)/f'round-{rnd}-activity.pt',weights_only=False);cache=tuple(torch.cat([a,b]) for a,b in zip(cache,part))
  write_json(OUT/'progress.json',dict(phase='label_mines',seed=seed,positions=len(rows)))
  mines=torch.zeros((len(rows),256),dtype=torch.bool)
  for i,r in enumerate(rows):
   info=info_from(r);assert info.safe
   for n in info.mines:mines[i,n//r['size']*16+n%r['size']]=True
  assert not (mines&cache[3]).any() and not (mines&~cache[1]).any();torch.save(mines,out/'mine-labels.pt')
  with (out/'dataset.pkl').open('wb') as f:pickle.dump(dict(train=rows),f)
  write_json(out/'games.json',games);write_json(out/'manifest.json',dict(seed=seed,parent=str(parent(seed)/'best-expanded.pt'),positions=len(rows),parent_step=saved['step'],updates=1500))
  history=[];metrics={}
  for name in ('control','priority'):
   head=ActivityHead(True);head.load_state_dict(saved['head']);opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(saved['optimizer']));rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(saved['rng'])
   best=checkpoint(head,opt,rng,0,validate(head,valid),parent_step=saved['step']);save(out/f'best-{name}.pt',best)
   hard,safe=mistakes(head,cache,mines);metrics[name]=dict(initial_known_mines=len(hard),initial_safe=safe,positions=len(rows));sampling=0
   for step in range(1,1501):
    if (step-1)%250==0:
     hard,safe=mistakes(head,cache,mines);np.save(out/f'hard-{name}-{step-1}.npy',hard)
     write_json(OUT/'progress.json',dict(phase='training',seed=seed,variant=name,step=step-1,known_mine_errors=len(hard)))
    ix=[int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(32)]
    if name=='priority' and len(hard):jx=np.concatenate([rng.integers(len(rows),size=16),rng.choice(hard,size=16)]);sampling+=16
    else:jx=rng.integers(len(rows),size=32)
    a,l,c,y=[torch.cat([v[ix],w[jx]]) for v,w in zip(oldcache,cache)]
    opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
    if step%250==0:
     value=validate(head,valid);state=checkpoint(head,opt,rng,step,value,parent_step=saved['step']);save(out/f'latest-{name}.pt',state)
     if value<best['validation_loss']:best=state;save(out/f'best-{name}.pt',best)
     rec=dict(seed=seed,variant=name,step=step,validation_loss=value,selected_step=best['step'],known_mine_errors_at_refresh=len(hard));history.append(rec);print(rec,flush=True)
   head.load_state_dict(best['head']);hard,safe=mistakes(head,cache,mines);metrics[name].update(selected_known_mines=len(hard),selected_safe=safe,selected_step=best['step'],prioritized_draws=sampling)
  write_json(out/'fit-metrics.json',metrics);write_json(out/'history.json',history);write_json(out/'trained.json',dict(trained=True))
  del cache,mines,rows
 del oldcache,valid
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 results={}
 for seed in SEEDS:
  out=target(seed);results[str(seed)]={}
  for name in ('baseline','control','priority'):
   path=parent(seed)/'best-expanded.pt' if name=='baseline' else out/f'best-{name}.pt';saved=torch.load(path,weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head'])
   write_json(OUT/'progress.json',dict(phase='evaluation',seed=seed,variant=name));rr=evaluate_games(MemoPolicy(memo,head),games);write_json(out/f'{name}-games.json',rr)
   results[str(seed)][name]=dict(wins=sum(r['won'] for r in rr),n=len(rr),selected_step=0 if name=='baseline' else saved['step']);write_json(OUT/'results.json',results);print(seed,name,results[str(seed)][name],flush=True)
  write_json(out/'completed.json',dict(completed=True))
 assert all(digest(p)==v for p,v in hashes.items());write_json(OUT/'verification.json',dict(original_hashes_unchanged=True,teacher_actions=0));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
