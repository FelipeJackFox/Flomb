"""Paired RF5/RF9 continuation of adapted DAgger heads on identical cached data."""
import copy,hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.wide_reader import make_head
from experiments.repeat_adapted_dagger import origin,SEEDS
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.compare_input_representation import reserve
from experiments.capacity_probe import write_json
from experiments.expand_dagger_experience import features
from experiments.extend_spatial_decoder import validate
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
OUT=Path('runs/wide-reader-001')
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 paths=[BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl')]
 for s in SEEDS:paths.extend([origin(s)/'best-dagger.pt',origin(s)/'dataset.pkl',origin(s)/'train.pt',origin(s)/'holdout.pt'])
 hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(seeds=SEEDS,updates=1500,batch=64,parameters=12769,lr=.001,clipping=5,mixture='32 balanced original +32 fixed inherited DAgger',selection='min original holdout at0 and every250, earliest tie',primary='wide minus local, paired final boards',intervention='only second conv dilation/padding1 to3, RF5 to9; inherited weights and Adam, changed initial function',hashes=hashes))
 src=OUT/'source';src.mkdir()
 for p in [Path(__file__),Path('experiments/wide_reader.py'),Path('experiments/test_wide_reader.py')]:shutil.copy2(p,src/p.name)
 games=reserve();write_json(OUT/'games.json',games)
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)]
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False)
 selections={};pairing={}
 for seed in SEEDS:
  parent=torch.load(origin(seed)/'best-dagger.pt',weights_only=False);assert parent['step']>1500
  brain.encoder.load_state_dict(parent['encoder']);frozen=tensor_hash(brain.state_dict());memo=ActivityMemo(brain)
  write_json(OUT/'progress.json',dict(phase='cache_inherited_experience',seed=seed))
  rows=pickle.loads((origin(seed)/'dataset.pkl').read_bytes())['train'];experience=features(memo,rows);torch.save(experience,OUT/f'experience-{seed}.pt')
  train=torch.load(origin(seed)/'train.pt',weights_only=False);valid=torch.load(origin(seed)/'holdout.pt',weights_only=False)
  check=make_head();check.load_state_dict(parent['head']);assert abs(validate(check,valid)-parent['validation_loss'])<1e-5
  for arm,dilation in [('local',1),('wide',3)]:
   head=make_head(dilation);head.load_state_dict(parent['head']);opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(parent['optimizer']));rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['rng']);draws=hashlib.sha256();best=float('inf');history=[];started=time.monotonic()
   assert tensor_hash(head.state_dict())==tensor_hash(parent['head']) and sum(p.numel() for p in head.parameters())==12769
   for step in range(1501):
    if step:
     ix=np.array([int(rng.choice(groups[((parent['step']+step)*64+j)%len(groups)])) for j in range(32)],np.int64);jx=rng.integers(len(rows),size=32,dtype=np.int64);draws.update(ix.tobytes()+jx.tobytes())
     a,l,c,y=[torch.cat([u[ix],v[jx]]) for u,v in zip(train,experience)];opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
    if step%250==0:
     value=validate(head,valid);state=dict(head=copy.deepcopy(head.state_dict()),encoder=parent['encoder'],optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,validation_loss=value,dilation=dilation)
     if value<best:best=value;chosen=step;torch.save(state,OUT/f'best-{seed}-{arm}.pt')
     torch.save(state,OUT/f'latest-{seed}-{arm}.pt');record=dict(phase='training',seed=seed,arm=arm,step=step,validation_loss=value,chosen=chosen,seconds=time.monotonic()-started);history.append(record);write_json(OUT/'progress.json',record);write_json(OUT/f'history-{seed}-{arm}.json',history);print(record,flush=True)
   key=f'{seed}-{arm}';selections[key]=dict(step=chosen,validation_loss=best,dilation=dilation,sha256=digest(OUT/f'best-{key}.pt'));pairing[key]=draws.hexdigest()
  assert pairing[f'{seed}-local']==pairing[f'{seed}-wide'] and frozen==tensor_hash(brain.state_dict())
 write_json(OUT/'pairing.json',pairing);write_json(OUT/'selection.json',selections);seal=digest(OUT/'selection.json');write_json(OUT/'selection-seal.json',dict(sha256=seal));results={}
 for seed in SEEDS:
  parent=torch.load(origin(seed)/'best-dagger.pt',weights_only=False);brain.encoder.load_state_dict(parent['encoder']);memo=ActivityMemo(brain)
  for arm in ('baseline','local','wide'):
   key=f'{seed}-{arm}';saved=parent if arm=='baseline' else torch.load(OUT/f'best-{key}.pt',weights_only=False);head=make_head(1 if arm=='baseline' else saved['dilation']);head.load_state_dict(saved['head'])
   write_json(OUT/'progress.json',dict(phase='evaluation',candidate=key,games=500));rows=evaluate_games(MemoPolicy(memo,head),games);write_json(OUT/f'{key}-games.json',rows);played=[r for r in rows if not r['automatic']]
   results[key]=dict(wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played),known_mine_choices=sum(r['known_mine_choices'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played));write_json(OUT/'results.json',results);print(key,results[key],flush=True)
 assert all(digest(p)==sha for p,sha in hashes.items()) and digest(OUT/'selection.json')==seal
 write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen_during_training=True,paired_batches=True,initial_weights_equal=True,parameter_count_equal=True,selection_sealed=True));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
