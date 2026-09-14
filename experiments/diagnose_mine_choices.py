"""Read-only diagnosis of known-mine clicks; no training or checkpoint changes."""
import json,pickle,hashlib,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from solver import analyze,deduce
from minesweeper import Minesweeper
from experiments.expand_dagger_experience import SEEDS,target,origin,BASE,CACHE,digest
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import arrays,write_json
from experiments.train import visible,public_context
from experiments.backbone import encode_visible
OUT=Path('runs/mine-choice-diagnostic-001')
def proof(board,size,mines,action,info):
 b=np.asarray(board).reshape(16,16)[:size,:size].flatten();hidden=set(map(int,np.flatnonzero(b==-1)));local=[];witness=[]
 for i in np.flatnonzero(b>=0):
  r,c=divmod(int(i),size);cells={rr*size+cc for rr in range(max(0,r-1),min(size,r+2)) for cc in range(max(0,c-1),min(size,c+2)) if (rr,cc)!=(r,c)}&hidden
  if cells:
   local.append((cells,int(b[i])))
   if action in cells and int(b[i])==len(cells):witness.append(dict(clue=int(i),number=int(b[i]),hidden_neighbors=sorted(cells)))
 if action not in info.mines:return 'not_proven_mine',[]
 if witness:return 'single_clue',witness
 if action in deduce(local)[1]:return 'local_subset',[]
 if action in deduce([(hidden,mines),*local])[1]:return 'global_count',[]
 return 'exact_enumeration',[]
def info_from(row):
 s=row['size'];b=row['board'].reshape(16,16)[:s,:s].flatten();return analyze(np.eye(10,dtype=np.float32)[b+1].flatten(),b==-1,s,row['mines'])
@torch.no_grad()
def cache_audit(seed):
 out=target(seed);saved=torch.load(out/'best-expanded.pt',weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head']);head.eval();last=(saved['step']+749)//750
 sets=[('inherited',pickle.loads((origin(seed)/'dataset.pkl').read_bytes())['train'],torch.load(out/'inherited.pt',weights_only=False))]
 for rnd in range(1,4):sets.append((f'round{rnd}',pickle.loads((out/f'round-{rnd}-positions.pkl').read_bytes()),torch.load(out/f'round-{rnd}-activity.pt',weights_only=False)))
 validrows=pickle.loads((BASE/'dataset.pkl').read_bytes())['holdout'];sets.append(('validation',validrows,torch.load(CACHE/'holdout.pt',weights_only=False)))
 results={};examples=[]
 for name,rows,cache in sets:
  ctr=Counter();ctr['positions']=len(rows);used=name=='inherited' or (name.startswith('round') and int(name[-1])<=last)
  for start in range(0,len(rows),128):
   a,l,c,y=[v[start:start+128] for v in cache];z=head(a,c).masked_fill(~l,-1e9);acts=z.argmax(1);p=z.softmax(1)
   ctr['safe_choices']+=int(y[torch.arange(len(acts)),acts].sum())
   for j,action in enumerate(acts.tolist()):
    r=rows[start+j]
    if y[j,action]:continue
    info=info_from(r);native=action//16*r['size']+action%16
    if native not in info.mines:ctr['uncertain_choices_with_safe']+=1;continue
    typ,wit=proof(r['board'],r['size'],r['mines'],native,info);ctr['known_mine_choices']+=1;ctr[typ]+=1
    if len(examples)<12 and typ=='single_clue':
     safe_ix=np.flatnonzero(y[j].numpy());best_safe=int(safe_ix[int(z[j,safe_ix].argmax())]);examples.append(dict(seed=seed,split=name,available_to_selected_model=used,board=r['board'].reshape(16,16)[:r['size'],:r['size']].tolist(),size=r['size'],action=native,best_safe=(best_safe//16)*r['size']+best_safe%16,chosen_probability=float(p[j,action]),score_margin=float(z[j,action]-z[j,best_safe]),witness=wit))
  results[name]=dict(ctr,available_to_selected_model=used)
 write_json(OUT/f'cached-{seed}.json',results);write_json(OUT/f'examples-{seed}.json',examples);return results
@torch.no_grad()
def replay():
 seed=SEEDS[0];saved=torch.load(target(seed)/'best-expanded.pt',weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head']);head.eval()
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 games=json.loads((target(seed)/'games.json').read_text());expected={r['seed']:r for r in json.loads((target(seed)/'expanded-games.json').read_text())};ctr=Counter();records=[];finished=[]
 for start in range(0,len(games),16):
  active=[(Minesweeper(g['seed'],7,7),g,Counter()) for g in games[start:start+16]]
  while active:
   x,legal=encode_visible([visible(e) for e,g,k in active]);c=torch.tensor(np.array([public_context(7,7) for _ in active]));z=head(memo.get(torch.from_numpy(x),c),c).masked_fill(~torch.from_numpy(legal),-1e9);acts=z.argmax(1).tolist();remaining=[]
   for j,((env,g,k),action) in enumerate(zip(active,acts)):
    b=visible(env);native=action//16*7+action%16;info=analyze(env.observation(),env.legal_mask(),7,7);safe=bool(info.safe);known=native in info.mines
    ctr['clicks']+=1;ctr['with_safe' if safe else 'without_safe']+=1;k['clicks']+=1;k['known_mine_choices']+=int(known);k['safe_opportunities']+=int(safe);k['safe_choices']+=int(native in info.safe)
    if known:
     typ,wit=proof(b,7,7,native,info);ctr['known_mine_choices']+=1;ctr['known_mine_with_safe' if safe else 'known_mine_without_safe']+=1;ctr[typ]+=1
     alternatives=[i for i in np.flatnonzero(env.legal_mask()) if i not in info.mines];assert alternatives
     records.append(dict(game_seed=g['seed'],board=b.reshape(16,16)[:7,:7].tolist(),action=native,has_safe=safe,proof=typ,witness=wit,chosen_logit=float(z[j,action]),non_mine_alternatives=len(alternatives)))
    env.step(native)
    if env.done:
     old=expected[g['seed']];assert bool(env.won)==old['won']
     for key in ('clicks','known_mine_choices','safe_opportunities','safe_choices'):assert k[key]==old[key],(key,k,old)
     ctr['wins']+=int(env.won);ctr['death_with_safe']+=int(not env.won and safe);finished.append(g['seed'])
    else:remaining.append((env,g,k))
   active=remaining
  write_json(OUT/'progress.json',dict(phase='replay',games=len(finished),target=500));print('replay',len(finished),flush=True)
 assert len(finished)==500;write_json(OUT/'replay.json',dict(ctr));write_json(OUT/'mine-events.json',records);op.close();return dict(ctr)
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise SystemExit('Preserve diagnostic')
 paths=[target(s)/'best-expanded.pt' for s in SEEDS]+[BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')];hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(hashes=hashes,scope='cached fit all seeds; exact replay first seed; no training',benchmark_status='previous final benchmark now inspected for diagnosis; future interventions require new final games'))
 (OUT/'source').mkdir();shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
 results={}
 for seed in SEEDS:
  write_json(OUT/'progress.json',dict(phase='cached_fit',seed=seed));results[str(seed)]=cache_audit(seed);print('cached',seed,flush=True)
 write_json(OUT/'cached-summary.json',results);replay()
 assert all(digest(p)==v for p,v in hashes.items());write_json(OUT/'verification.json',dict(hashes_unchanged=True,replay_matches_prior_game_metrics=True));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
if __name__=='__main__':main()
