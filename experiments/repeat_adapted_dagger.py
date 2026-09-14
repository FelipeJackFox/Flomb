"""Replicate adapted-encoder DAgger; two new seeds primary, discovery secondary."""
import json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments import adapted_dagger as training
from experiments.train_joint_interface import BASE,digest,tensor_hash,frozen_hash
from experiments.adapted_dagger import reserve
from experiments.capacity_probe import write_json
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games
OUT=Path('runs/adapted-dagger-consistency-001');SEEDS=(20261002,20261003,20261004)
def origin(seed):return Path('runs/adapted-dagger-001' if seed==SEEDS[0] else f'runs/adapted-dagger-{seed}')
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 protected=[BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 for seed in SEEDS:protected.append(Path(f'runs/joint-extended-{seed}/best-joint.pt'))
 for arm in ('control','dagger'):protected.append(origin(SEEDS[0])/f'best-{arm}.pt')
 hashes={str(p):digest(p) for p in protected}
 write_json(OUT/'manifest.json',dict(seeds=SEEDS,new_seeds=list(SEEDS[1:]),reused_seed=SEEDS[0],updates=2250,batch=64,microbatch=16,hashes=hashes,primary='mean paired dagger minus control on the two new seeds; all three secondary',selection='same original holdout rule; seal every checkpoint before any final games'))
 src=OUT/'source';src.mkdir()
 for p in (Path(__file__),Path('experiments/adapted_dagger.py'),Path('experiments/spatial_dagger.py')):shutil.copy2(p,src/p.name)
 splits=reserve();games=splits['games'];write_json(OUT/'games.json',games);write_json(OUT/'collection-games.json',splits['collection'])
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games,collection=splits['collection']),f)
 for seed in SEEDS[1:]:
  write_json(OUT/'progress.json',dict(phase='training',seed=seed,detail_file=str(origin(seed)/'progress.json')))
  training.main(seed=seed,out=origin(seed),splits=splits,train_only=True)
 chosen={}
 for seed in SEEDS:
  for arm in ('control','dagger'):
   p=origin(seed)/f'best-{arm}.pt';chosen[f'{seed}-{arm}']=dict(path=str(p),sha256=digest(p))
 write_json(OUT/'selection.json',chosen);seal=digest(OUT/'selection.json');write_json(OUT/'selection-seal.json',dict(sha256=seal))
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
 brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);fixed=frozen_hash(brain)
 encoder={k:v.clone() for k,v in brain.encoder.state_dict().items()};memos={};seen={};results={}
 for seed in SEEDS:
  for arm in ('baseline','control','dagger'):
   key=f'{seed}-{arm}'
   saved=torch.load(Path(f'runs/joint-extended-{seed}/best-joint.pt') if arm=='baseline' else origin(seed)/f'best-{arm}.pt',weights_only=False)
   state=saved['encoder'];eid=tensor_hash(state);weightid=eid+tensor_hash(saved['head'])
   if weightid in seen:
    prior=seen[weightid];rows=json.loads((OUT/f'{prior}-games.json').read_text());result=dict(results[prior],duplicate_of=prior)
   else:
    brain.encoder.load_state_dict(state);head=ActivityHead(True);head.load_state_dict(saved['head']);memo=memos.setdefault(eid,ActivityMemo(brain))
    write_json(OUT/'progress.json',dict(phase='evaluation',candidate=key,games=500))
    rows=evaluate_games(MemoPolicy(memo,head),games);played=[r for r in rows if not r['automatic']]
    result=dict(wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played),known_mine_choices=sum(r['known_mine_choices'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played));seen[weightid]=key
   write_json(OUT/f'{key}-games.json',rows);results[key]=result;write_json(OUT/'results.json',results);print(key,result,flush=True)
 assert fixed==frozen_hash(brain) and digest(OUT/'selection.json')==seal
 assert all(digest(p)==sha for p,sha in hashes.items())
 assert all(digest(r['path'])==r['sha256'] for r in chosen.values())
 write_json(OUT/'verification.json',dict(originals_unchanged=True,selected_weights_sealed=True,frozen_brain_unchanged=True,unique_models_evaluated=len(seen)))
 write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
