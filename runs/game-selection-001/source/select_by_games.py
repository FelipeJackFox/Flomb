"""Frozen saved candidates: select on new validation games, test separately."""
import hashlib,json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments.prioritize_mine_errors import target as prior,parent,BASE,digest,SEEDS
from experiments.expand_dagger_experience import reserve
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.scaled_train import evaluate_games
from experiments.capacity_probe import write_json
OUT=Path('runs/game-selection-001')
def weights_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.contiguous().numpy().tobytes())
 return h.hexdigest()
def choose(keys,candidates,validation):
 # All candidates see identical games; automated openings excluded identically.
 return min(keys,key=lambda k:(-validation[k]['wins'],candidates[k]['step'],keys.index(k)))
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise SystemExit('Preserve existing experiment')
 candidates={};states={};arms={}
 for seed in SEEDS:
  key=f'{seed}-baseline';path=parent(seed)/'best-expanded.pt';saved=torch.load(path,weights_only=False)
  candidates[key]=dict(seed=seed,kind='baseline',path=str(path),step=0,validation_loss=saved['validation_loss'],weights_hash=weights_hash(saved['head']));states[key]=saved['head']
  for arm in ('control','priority'):
   keys=[key]
   for kind in ('best','latest'):
    name=f'{seed}-{arm}-{kind}';path=prior(seed)/f'{kind}-{arm}.pt';saved=torch.load(path,weights_only=False)
    candidates[name]=dict(seed=seed,kind=f'{arm}-{kind}',path=str(path),step=saved['step'],validation_loss=saved['validation_loss'],weights_hash=weights_hash(saved['head']));states[name]=saved['head'];keys.append(name)
   arms[f'{seed}-{arm}']=keys
 protected={r['path']:digest(r['path']) for r in candidates.values()}
 for p in [BASE/'training/retina_plastic-20260926.pt',BASE/'mapping.pkl',Path('runs/hybrid-001/checkpoint.pkl')]:protected[str(p)]=digest(p)
 write_json(OUT/'manifest.json',dict(candidates=candidates,arms=arms,protected_hashes=protected,selection='highest validation autonomous wins; ties fewer additional updates then fixed candidate order',validation_games=500,test_games=500,training=False))
 source=OUT/'source';source.mkdir()
 for p in [Path(__file__),Path('experiments/scaled_train.py'),Path('experiments/validate_spatial_decoder.py'),Path('experiments/spatial_decoder.py'),Path('solver.py'),Path('minesweeper.py')]:shutil.copy2(p,source/p.name)
 reserved=reserve();splits=dict(test=reserved['games'],validation=reserved['collection'][:500]);write_json(OUT/'games.json',splits['test']);write_json(OUT/'validation-games.json',splits['validation'])
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=splits['test'],holdout_games=splits['validation']),f)
 assert not {r['layout_hash'] for r in splits['test']}&{r['layout_hash'] for r in splits['validation']}
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval();memo=ActivityMemo(brain)
 seen={s:{} for s in splits};validation={};test={}
 def evaluate(key,split,table):
  identity=candidates[key]['weights_hash']
  if identity in seen[split]:table[key]=dict(seen[split][identity],duplicate_of=seen[split][identity]['candidate']);return
  write_json(OUT/'progress.json',dict(phase=split,candidate=key,unique_completed=len(seen[split])));head=ActivityHead(True);head.load_state_dict(states[key]);head.eval()
  rows=evaluate_games(MemoPolicy(memo,head),splits[split]);file=f'{split}-{key}.json';write_json(OUT/file,rows);played=[r for r in rows if not r['automatic']]
  entry=dict(candidate=key,rows_file=file,wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played));table[key]=entry;seen[split][identity]=entry
  write_json(OUT/f'{split}-results.json',table);print(split,key,entry,flush=True)
 for key in candidates:evaluate(key,'validation',validation)
 write_json(OUT/'validation-results.json',validation)
 choices={arm:dict(loss=keys[1],games=choose(keys,candidates,validation),baseline=keys[0]) for arm,keys in arms.items()}
 write_json(OUT/'selection.json',choices);seal=digest(OUT/'selection.json');write_json(OUT/'selection-seal.json',dict(sha256=seal));print('SELECTED',choices,flush=True)
 keys=list(dict.fromkeys(k for c in choices.values() for k in c.values()))
 for key in keys:evaluate(key,'test',test)
 write_json(OUT/'test-results.json',test)
 assert digest(OUT/'selection.json')==seal
 assert all(digest(p)==sha for p,sha in protected.items())
 write_json(OUT/'verification.json',dict(original_hashes_unchanged=True,selection_unchanged_during_test=True,unique_validation_models=len(seen['validation']),unique_test_models=len(seen['test']),memo_hits=memo.hits,memo_misses=memo.misses))
 write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
