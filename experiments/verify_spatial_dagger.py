"""Independent split, teacher-label, preservation and checkpoint-resume audit."""
import copy,json,pickle
from pathlib import Path
import numpy as np
import torch
from experiments.spatial_dagger import OUT
from experiments.extend_spatial_decoder import digest,CACHE
from experiments.spatial_decoder import ActivityHead
from experiments.expressive_models import equivalent_loss
from experiments.capacity_probe import write_json
from solver import analyze

def main():
 torch.set_num_threads(1)
 assert all((OUT/f'best-{name}.pt').exists() for name in ('control','dagger'))
 games=json.loads((OUT/'games.json').read_text());collection=json.loads((OUT/'collection-games.json').read_text())
 test={r['layout_hash'] for r in games};train={r['layout_hash'] for r in collection};assert len(test)==500 and len(train)==900 and not test&train
 prior=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent==OUT:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):prior.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/games.json'):
  if p.parent!=OUT:prior.update(r['layout_hash'] for r in json.loads(p.read_text()))
 assert not prior&(test|train)
 rows=pickle.loads((OUT/'dataset.pkl').read_bytes())['train']
 for r in rows:
  assert r['layout_hash'] in train
  b=r['board'].reshape(16,16)[:7,:7].flatten();legal=b==-1;obs=np.eye(10,dtype=np.float32)[b+1].flatten();info=analyze(obs,legal,7,7)
  expected=np.zeros(256,bool)
  for i in info.safe:expected[i//7*16+i%7]=True
  assert expected.any();np.testing.assert_array_equal(expected,r['labels'])
  action=r['learner_action'];native=action//16*7+action%16
  assert legal[native] and r['learner_safe']==bool(native in info.safe)
 manifest=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
 a,l,c,y=[v[:64] for v in torch.load(CACHE/'train.pt',weights_only=False)]
 for name in ('control','dagger'):
  for kind in ('best','latest'):
   state=torch.load(OUT/f'{kind}-{name}.pt',weights_only=False)
   assert all(torch.isfinite(v).all() for v in state['head'].values())
   hs=[]
   for _ in range(2):
    h=ActivityHead(True);h.load_state_dict(state['head']);opt=torch.optim.Adam(h.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(state['optimizer']))
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng']);ix=rng.integers(64,size=64)
    opt.zero_grad();loss=equivalent_loss(h(a[ix],c[ix]),l[ix],y[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True);opt.step();hs.append(h)
   for k,v in hs[0].state_dict().items():torch.testing.assert_close(v,hs[1].state_dict()[k],atol=0,rtol=0)
 write_json(OUT/'independent_verification.json',dict(layouts_disjoint=True,public_labels_verified=len(rows),original_hashes_unchanged=True,four_checkpoint_resume_replays_exact=True))
 print('PASS',len(rows),'public labels, disjoint splits, four exact checkpoint replay pairs')
if __name__=='__main__':main()
