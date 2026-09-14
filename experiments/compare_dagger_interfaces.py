"""Matched historical collection layouts: DAgger with/without interface adaptation."""
import json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments import adapted_dagger as training
from experiments.train_joint_interface import BASE,digest,tensor_hash,frozen_hash
from experiments.compare_input_representation import reserve
from experiments.repeat_adapted_dagger import origin as adapted,SEEDS
from experiments.capacity_probe import write_json
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games
OUT=Path('runs/dagger-interface-comparison-001')
def unadapted(seed):return Path(f'runs/unadapted-dagger-{seed}')
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 protected=[BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 for s in SEEDS:protected.extend([Path(f'runs/joint-extended-{s}/best-control.pt'),adapted(s)/'best-dagger.pt',adapted(s)/'collection-games.json'])
 hashes={str(p):digest(p) for p in protected}
 write_json(OUT/'manifest.json',dict(seeds=SEEDS,hashes=hashes,updates=2250,rounds=3,collection_games_per_seed=900,primary='adapted DAgger minus unadapted DAgger, paired by final board',scope='pipeline comparison; matched prior head update budgets, different selected encoder/head/RNG states; not isolated encoder ablation',collection='reuse exactly each adapted counterparts historical900 layouts, not final test',selection='same historical holdout loss rule; all candidates sealed before final test'))
 src=OUT/'source';src.mkdir()
 for p in (Path(__file__),Path('experiments/adapted_dagger.py')):shutil.copy2(p,src/p.name)
 games=reserve();write_json(OUT/'games.json',games)
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
 for s in SEEDS:
  collection=json.loads((adapted(s)/'collection-games.json').read_text());assert not {r['layout_hash'] for r in collection}&{r['layout_hash'] for r in games}
  write_json(OUT/'progress.json',dict(phase='training',seed=s,detail_file=str(unadapted(s)/'progress.json')))
  training.main(out=unadapted(s),seed=s,splits=dict(games=games,collection=collection),train_only=True,parent_path=Path(f'runs/joint-extended-{s}/best-control.pt'))
 selected={}
 for s in SEEDS:
  for arm,path in [('unadapted',unadapted(s)),('adapted',adapted(s))]:
   p=path/'best-dagger.pt';selected[f'{s}-{arm}']=dict(path=str(p),sha256=digest(p))
 write_json(OUT/'selection.json',selected);seal=digest(OUT/'selection.json');write_json(OUT/'selection-seal.json',dict(sha256=seal))
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);fixed=frozen_hash(brain);memos={};results={}
 for key,entry in selected.items():
  saved=torch.load(entry['path'],weights_only=False);brain.encoder.load_state_dict(saved['encoder']);head=ActivityHead(True);head.load_state_dict(saved['head']);eid=tensor_hash(saved['encoder']);memo=memos.setdefault(eid,ActivityMemo(brain))
  write_json(OUT/'progress.json',dict(phase='evaluation',candidate=key,games=500));rows=evaluate_games(MemoPolicy(memo,head),games);write_json(OUT/f'{key}-games.json',rows);played=[r for r in rows if not r['automatic']]
  results[key]=dict(wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played),known_mine_choices=sum(r['known_mine_choices'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played));write_json(OUT/'results.json',results);print(key,results[key],flush=True)
 assert frozen_hash(brain)==fixed and digest(OUT/'selection.json')==seal and all(digest(p)==sha for p,sha in hashes.items())
 assert all(digest(r['path'])==r['sha256'] for r in selected.values())
 write_json(OUT/'verification.json',dict(originals_intact=True,fixed_connectome=True,selection_sealed=True));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
