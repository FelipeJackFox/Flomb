"""Independent shared split, public label, frozen-input and resume checks."""
import copy,json,pickle
from pathlib import Path
import numpy as np
import torch
from solver import analyze
from experiments.extend_spatial_decoder import CACHE,digest
from experiments.spatial_decoder import ActivityHead
from experiments.expressive_models import equivalent_loss
from experiments.capacity_probe import write_json
from experiments.validate_spatial_decoder import SEEDS
P=Path('runs/spatial-dagger-consistency-001')
def main(seeds=SEEDS[1:]):
 torch.set_num_threads(1)
 peers={P,*[Path(f'runs/spatial-dagger-{s}') for s in SEEDS[1:]]}
 test={r['layout_hash'] for r in json.loads((P/'games.json').read_text())};assert len(test)==500
 collection=json.loads(Path('runs/spatial-dagger-001/collection-games.json').read_text());train={r['layout_hash'] for r in collection};assert len(train)==900 and not test&train
 prior=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent in peers:continue
  for rr in pickle.loads(p.read_bytes()).values():
   if isinstance(rr,list):prior.update(r['layout_hash'] for r in rr if isinstance(r,dict) and 'layout_hash' in r)
 for pat in ('*/games.json','*/collection-games.json'):
  for p in Path('runs').glob(pat):
   if p.parent not in peers:prior.update(r['layout_hash'] for r in json.loads(p.read_text()))
 assert not test&prior
 counts={};a,l,c,y=[v[:64] for v in torch.load(CACHE/'train.pt',weights_only=False)]
 for s in seeds:
  out=Path(f'runs/spatial-dagger-{s}');assert (out/'completed.json').exists()
  assert json.loads((out/'collection-games.json').read_text())==collection
  assert {r['layout_hash'] for r in json.loads((out/'games.json').read_text())}==test
  rr=pickle.loads((out/'dataset.pkl').read_bytes())['train'];counts[s]=len(rr)
  for r in rr:
   assert r['layout_hash'] in train
   b=r['board'].reshape(16,16)[:7,:7].flatten();legal=b==-1
   info=analyze(np.eye(10,dtype=np.float32)[b+1].flatten(),legal,7,7);expected=np.zeros(256,bool)
   for i in info.safe:expected[i//7*16+i%7]=True
   assert expected.any();np.testing.assert_array_equal(expected,r['labels'])
   action=r['learner_action'];native=action//16*7+action%16;assert legal[native] and r['learner_safe']==bool(native in info.safe)
  for name in ('control','dagger'):
   for kind in ('best','latest'):
    state=torch.load(out/f'{kind}-{name}.pt',weights_only=False);hs=[]
    assert all(torch.isfinite(v).all() for v in state['head'].values())
    for _ in range(2):
     h=ActivityHead(True);h.load_state_dict(state['head']);opt=torch.optim.Adam(h.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(state['optimizer']))
     rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng']);ix=rng.integers(64,size=64)
     opt.zero_grad();loss=equivalent_loss(h(a[ix],c[ix]),l[ix],y[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True);opt.step();hs.append(h)
    for k,v in hs[0].state_dict().items():torch.testing.assert_close(v,hs[1].state_dict()[k],atol=0,rtol=0)
  assert all(digest(p)==sha for p,sha in json.loads((out/'manifest.json').read_text())['hashes'].items())
 assert all(digest(p)==sha for p,sha in json.loads((P/'manifest.json').read_text())['hashes'].items())
 write_json(P/('independent_verification.json' if len(seeds)==2 else f'independent_verification-{seeds[0]}.json'),dict(test_prior_overlap=0,identical_collection_layouts=True,public_labels_verified=counts,original_hashes_unchanged=True,checkpoint_replay_pairs_exact=4*len(seeds)))
 print('PASS',counts)
if __name__=='__main__':main()
