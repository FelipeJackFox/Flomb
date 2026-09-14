"""Two new head seeds; reuse training layouts, fresh common held-out games."""
import json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments import spatial_dagger as d
from experiments.extend_spatial_decoder import digest,BASE
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy,SEEDS
from experiments.spatial_decoder import ActivityHead
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.scaled_train import evaluate_games
from experiments.scaled_data import identity
from experiments.capacity_probe import write_json
from minesweeper import Minesweeper
OUT=Path('runs/spatial-dagger-consistency-001');PILOT=Path('runs/spatial-dagger-001')
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise SystemExit('Preserve existing run')
 paths=[PILOT/'best-dagger.pt',PILOT/'best-control.pt',PILOT/'collection-games.json',*[Path('runs/spatial-extended-001')/f'best-{s}.pt' for s in SEEDS],BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
 hashes={str(p):digest(p) for p in paths}
 write_json(OUT/'manifest.json',dict(seeds=SEEDS,reused_seed=SEEDS[0],new_seeds=SEEDS[1:],collection='same 900 layouts as first seed, intentional train reuse',selection='same old holdout, min loss including baseline',hashes=hashes))
 (OUT/'source').mkdir()
 for f in [Path(__file__),Path(d.__file__)]:shutil.copy2(f,OUT/'source'/f.name)
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for pattern in ('*/games.json','*/collection-games.json'):
  for p in Path('runs').glob(pattern):used.update(r['layout_hash'] for r in json.loads(p.read_text()))
 games=[];seed=6000700000
 while len(games)<500:
  key=identity(Minesweeper(seed,7,7))
  if key not in used:games.append(dict(seed=seed,size=7,mines=7,layout_hash=key));used.add(key)
  seed+=1
 write_json(OUT/'games.json',games);collection=json.loads((PILOT/'collection-games.json').read_text())
 assert not {r['layout_hash'] for r in games}&{r['layout_hash'] for r in collection}
 write_json(OUT/'split_verification.json',dict(final_games=500,final_prior_overlap=0,collection_layouts=900,intentional_collection_reuse=True))
 results={}
 for seed in SEEDS[1:]:
  sub=Path(f'runs/spatial-dagger-{seed}');write_json(OUT/'progress.json',dict(phase='replication',seed=seed,details=str(sub/'progress.json')))
  d.main(sub,seed,dict(games=games,collection=collection))
  results[str(seed)]=json.loads((sub/'results.json').read_text());write_json(OUT/'results.json',results)
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 first={}
 for name in ('baseline','control','dagger'):
  p=Path('runs/spatial-extended-001')/f'best-{SEEDS[0]}.pt' if name=='baseline' else PILOT/f'best-{name}.pt'
  saved=torch.load(p,weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head'])
  write_json(OUT/'progress.json',dict(phase='evaluate_reused_seed',variant=name,seed=SEEDS[0]));rows=evaluate_games(MemoPolicy(memo,head),games);write_json(OUT/f'{name}-games.json',rows)
  first[name]=dict(wins=sum(r['won'] for r in rows),n=len(rows),selected_step=0 if name=='baseline' else saved['step']);print('reused',name,first[name],flush=True)
 results[str(SEEDS[0])]=first;write_json(OUT/'results.json',results)
 assert all(digest(p)==h for p,h in hashes.items())
 write_json(OUT/'verification.json',dict(protected_hashes_unchanged=True));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
