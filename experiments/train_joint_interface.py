"""Paired 750-update pilot: adapt encoder+reader versus frozen encoder control."""
import copy,hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.joint_interface import JointInterface
from experiments.compare_input_representation import reserve
from experiments.capacity_probe import arrays,write_json
from experiments.extend_spatial_decoder import BASE,CACHE,digest,validate
from experiments.spatial_decoder import ActivityHead,activity_map
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.expressive_models import equivalent_loss
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games

OUT=Path('runs/joint-interface-001');PARENT=Path('runs/input-representation-001/best-brain-20261002.pt');UPDATES=750

def tensor_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().contiguous().numpy().tobytes())
 return h.hexdigest()

def frozen_hash(brain):return tensor_hash({k:v for k,v in brain.state_dict().items() if not k.startswith('encoder.')})

def save(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

@torch.no_grad()
def validation(model,rows):
 total=0.
 for i in range(0,len(rows),16):
  x,l,c,y=arrays(rows[i:i+16]);total+=float(equivalent_loss(model(x,c),l,y))*len(x)
 return total/len(rows)

def main(seed=20261002,output=None,games_path=None,train_only=False,updates=750,resume_from=None):
 global OUT,PARENT,UPDATES
 UPDATES=updates
 if output is not None:OUT=Path(output)
 PARENT=Path(f'runs/input-representation-001/best-brain-{seed}.pt')
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 paths=[PARENT,BASE/'dataset.pkl',BASE/'mapping.pkl',BASE/'training/retina_plastic-20260926.pt',CACHE/'train.pt',CACHE/'holdout.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 if resume_from:
  resume_from=Path(resume_from)
  paths.extend(resume_from/f'{kind}-{arm}.pt' for kind in ('best','latest') for arm in ('control','joint'))
 hashes={str(p):digest(p) for p in paths}
 write_json(OUT/'manifest.json',dict(seed=seed,updates=UPDATES,batch=64,microbatch=16,head_lr=.001,encoder_lr=.0001,parent=str(PARENT),resume_from=str(resume_from) if resume_from else None,selection='minimum original holdout loss every250 including inherited best; earliest tie',scope='paired adaptation, pretrained brain and selected head; no new DAgger',hashes=hashes))
 src=OUT/'source';src.mkdir()
 for p in [Path(__file__),*[Path('experiments')/n for n in ('joint_interface.py','test_joint_interface.py','retina_policy.py','plastic_sparse.py','spatial_decoder.py','expressive_models.py','capacity_probe.py','scaled_train.py','validate_spatial_decoder.py')],Path('solver.py'),Path('minesweeper.py')]:shutil.copy2(p,src/p.name)
 games=reserve() if games_path is None else json.loads(Path(games_path).read_text());write_json(OUT/'games.json',games)
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());cache={s:torch.load(CACHE/f'{s}.pt',weights_only=False) for s in ('train','holdout')}
 for split in cache:
  _,l,c,y=arrays(data[split])
  for a,b in zip((l,c,y),cache[split][1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
 groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)]
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
 brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False)
 original_encoder=copy.deepcopy(brain.encoder.state_dict());fixed_before=frozen_hash(brain)
 old=torch.load(PARENT,weights_only=False);head=ActivityHead(True);head.load_state_dict(old['head']);model=JointInterface(brain,head)
 # Actual graph forward and accumulation parity before training.
 x,l,c,y=arrays(data['train'][:4])
 torch.testing.assert_close(model.activity(x,c),activity_map(brain,x,c),rtol=1e-5,atol=1e-5)
 model.zero_grad();equivalent_loss(model(x,c),l,y).backward()
 params=list(head.parameters())+list(brain.encoder.parameters());full=[p.grad.clone() for p in params]
 model.zero_grad()
 for i in (0,2):(equivalent_loss(model(x[i:i+2],c[i:i+2]),l[i:i+2],y[i:i+2])/2).backward()
 for p,g in zip(params,full):torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=2e-6)
 encoder_gradient=sum(float(p.grad.abs().sum()) for p in brain.encoder.parameters());assert encoder_gradient>0
 model.zero_grad();write_json(OUT/'preflight.json',dict(real_forward_parity=True,real_accumulation_gradient_parity=True,encoder_gradient_l1=encoder_gradient))
 # Equivalent initial cache validation, later mutable encoder must be recomputed.
 base_value=validate(head,cache['holdout']);online_value=validation(model,data['holdout']);assert abs(base_value-online_value)<1e-5
 selection={};pairing={}
 for arm in ('control','joint'):
  head.load_state_dict(old['head']);brain.encoder.load_state_dict(original_encoder)
  opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(old['optimizer']))
  if arm=='joint':opt.add_param_group(dict(params=list(brain.encoder.parameters()),lr=.0001))
  rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(old['rng']);draws=hashlib.sha256()
  best=float('inf');history=[];started=time.monotonic()
  start=0
  if resume_from:
   last=torch.load(resume_from/f'latest-{arm}.pt',weights_only=False);prior=torch.load(resume_from/f'best-{arm}.pt',weights_only=False)
   head.load_state_dict(last['head']);brain.encoder.load_state_dict(last['encoder']);opt.load_state_dict(last['optimizer']);rng.bit_generator.state=copy.deepcopy(last['rng'])
   assert tensor_hash(head.state_dict())==tensor_hash(last['head']) and tensor_hash(brain.encoder.state_dict())==tensor_hash(last['encoder'])
   for param_id,state in last['optimizer']['state'].items():
    for name,value in state.items():
     restored=opt.state_dict()['state'][param_id][name]
     if torch.is_tensor(value):torch.testing.assert_close(value,restored,rtol=0,atol=0)
     else:assert value==restored
   assert rng.bit_generator.state==last['rng'] and last['parent_step']==old['step']
   start=last['step']+1;assert start<=UPDATES
   best=prior['validation_loss'];chosen=prior['step'];history=json.loads((resume_from/f'history-{arm}.json').read_text())
   save(OUT/f'best-{arm}.pt',prior);save(OUT/f'latest-{arm}.pt',last)
   write_json(OUT/f'resume-{arm}.json',dict(source_step=last['step'],weights_optimizer_rng_exact=True,inherited_best_step=chosen))
  for step in range(start,UPDATES+1):
   if step:
    ix=np.array([int(rng.choice(groups[((old['step']+step)*64+j)%len(groups)])) for j in range(64)],np.int64);draws.update(ix.tobytes());opt.zero_grad();model.zero_grad()
    if arm=='control':
     a,ll,cc,yy=[v[ix] for v in cache['train']];loss=equivalent_loss(head(a,cc),ll,yy);loss.backward()
    else:
     for offset in range(0,64,16):
      xx,ll,cc,yy=arrays([data['train'][i] for i in ix[offset:offset+16]])
      loss=equivalent_loss(model(xx,cc),ll,yy)/4;loss.backward()
    # Clip head independently in both arms; encoder separately in joint arm.
    torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
    if arm=='joint':torch.nn.utils.clip_grad_norm_(brain.encoder.parameters(),5.,error_if_nonfinite=True)
    opt.step()
   if step%25==0:
    progress=dict(phase='training',arm=arm,step=step,target=UPDATES,seconds=time.monotonic()-started);write_json(OUT/'progress.json',progress);print(progress,flush=True)
   if step%250==0:
    value=base_value if step==0 else (validate(head,cache['holdout']) if arm=='control' else validation(model,data['holdout']))
    state=dict(head=copy.deepcopy(head.state_dict()),encoder=copy.deepcopy(brain.encoder.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,validation_loss=value,parent_step=old['step'])
    save(OUT/f'latest-{arm}.pt',state)
    if value<best:best=value;chosen=step;save(OUT/f'best-{arm}.pt',state)
    record=dict(step=step,validation_loss=value,chosen=chosen,seconds=time.monotonic()-started);history.append(record);write_json(OUT/f'history-{arm}.json',history);print(arm,record,flush=True)
  selection[arm]=dict(step=chosen,validation_loss=best);pairing[arm]=draws.hexdigest()
  assert frozen_hash(brain)==fixed_before
  if arm=='control':assert tensor_hash(brain.encoder.state_dict())==tensor_hash(original_encoder)
  else:assert tensor_hash(brain.encoder.state_dict())!=tensor_hash(original_encoder)
 assert pairing['control']==pairing['joint'];write_json(OUT/'pairing.json',pairing)
 write_json(OUT/'selection.json',selection);seal=digest(OUT/'selection.json');write_json(OUT/'selection-seal.json',dict(sha256=seal))
 if train_only:
  assert all(digest(p)==sha for p,sha in hashes.items())
  write_json(OUT/'training-verification.json',dict(original_hashes_unchanged=True,frozen_brain_parameters_unchanged=True,control_encoder_unchanged=True,joint_encoder_changed=True,paired_minibatches=True,cache_and_online_initial_validation_equal=True))
  write_json(OUT/'training-completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='awaiting_shared_evaluation'));op.close();return
 results={}
 for arm in ('baseline','control','joint'):
  saved=dict(head=old['head'],encoder=original_encoder) if arm=='baseline' else torch.load(OUT/f'best-{arm}.pt',weights_only=False)
  head.load_state_dict(saved['head']);brain.encoder.load_state_dict(saved['encoder']);brain.requires_grad_(False)
  memo=ActivityMemo(brain);policy=MemoPolicy(memo,head)
  x,_,c,_=arrays(data['holdout'][:4])
  with torch.no_grad():torch.testing.assert_close(policy(x,c),model(x,c),rtol=1e-5,atol=1e-5)
  write_json(OUT/'progress.json',dict(phase='evaluation',arm=arm,games=500))
  rows=evaluate_games(policy,games);write_json(OUT/f'{arm}-games.json',rows);played=[r for r in rows if not r['automatic']]
  results[arm]=dict(wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played),known_mine_choices=sum(r['known_mine_choices'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played))
  write_json(OUT/'results.json',results);print(arm,results[arm],flush=True)
 assert frozen_hash(brain)==fixed_before;assert all(digest(p)==sha for p,sha in hashes.items());assert digest(OUT/'selection.json')==seal
 write_json(OUT/'verification.json',dict(original_hashes_unchanged=True,frozen_brain_parameters_unchanged=True,control_encoder_unchanged=True,joint_encoder_changed=True,paired_minibatches=True,cache_and_online_initial_validation_equal=True,restored_online_and_memo_forward_equal=True,selection_seal_intact=True))
 write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
