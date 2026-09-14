import json,pickle
from pathlib import Path
import torch
from experiments.select_by_games import OUT,weights_hash
from experiments.expand_dagger_experience import digest
from experiments.scaled_data import identity
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json

def main():
 m=json.loads((OUT/'manifest.json').read_text());games=json.loads((OUT/'games.json').read_text());validation=json.loads((OUT/'validation-games.json').read_text())
 sets=[{r['layout_hash'] for r in rows} for rows in (games,validation)];assert all(len(s)==500 for s in sets) and not sets[0]&sets[1]
 for r in games+validation:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent==OUT:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for pat in ('*/games.json','*/collection-games.json','*/validation-games.json'):
  for p in Path('runs').glob(pat):
   if p.parent!=OUT:used.update(r['layout_hash'] for r in json.loads(p.read_text()))
 assert not (sets[0]|sets[1])&used
 for k,c in m['candidates'].items():assert weights_hash(torch.load(c['path'],weights_only=False)['head'])==c['weights_hash']
 sel=json.loads((OUT/'selection.json').read_text());val=json.loads((OUT/'validation-results.json').read_text());test=json.loads((OUT/'test-results.json').read_text());assert set(val)==set(m['candidates']);assert set(test)=={k for r in sel.values() for k in r.values()}
 for arm,keys in m['arms'].items():
  best=min(keys,key=lambda k:(-val[k]['wins'],m['candidates'][k]['step'],keys.index(k)))
  assert sel[arm]==dict(loss=keys[1],games=best,baseline=keys[0])
 for table,expected in ((val,validation),(test,games)):
  expected_ids=sorted((r['size'],r['seed'],r['layout_hash']) for r in expected);reference_auto=None
  for k,e in table.items():
   rows=json.loads((OUT/e['rows_file']).read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected_ids
   auto=[r['automatic'] for r in rows]
   if reference_auto is None:reference_auto=auto
   assert auto==reference_auto
   played=[r for r in rows if not r['automatic']];assert len(played)==e['n'] and sum(r['won'] for r in played)==e['wins']
   assert m['candidates'][k]['weights_hash']==m['candidates'][e['candidate']]['weights_hash']
 assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 assert all(digest(p)==sha for p,sha in m['protected_hashes'].items())
 write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=1000,split_overlap=0,prior_overlap=0,selection_rule_recomputed=True,selection_seal_intact=True,deduplicated_weights_verified=True,result_counts_verified=True,original_hashes_unchanged=True));print('PASS: independent split, selection, counts, deduplication, seal and original hashes')
if __name__=='__main__':main()
