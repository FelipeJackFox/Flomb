"""Independent labels, provenance, cache, optimizer and split verification."""
import copy,json,pickle
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from solver import analyze
from experiments.expand_dagger_experience import OUT,target,origin,SEEDS,BASE,CACHE,digest
from experiments.capacity_probe import arrays,write_json
from experiments.spatial_decoder import ActivityHead,activity_map
from experiments.expressive_models import equivalent_loss
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator

def same_rows(a,b):
 assert len(a)==len(b)
 for x,y in zip(a,b):
  assert x.keys()==y.keys()
  for k,v in x.items():
   if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,y[k])
   else:assert v==y[k]

def main(seeds=SEEDS):
 torch.set_num_threads(1)
 peers={OUT,*[target(s) for s in SEEDS]};games=json.loads((OUT/'games.json').read_text());collection=json.loads((OUT/'collection-games.json').read_text())
 test={r['layout_hash'] for r in games};train={r['layout_hash'] for r in collection};assert len(test)==500 and len(train)==1800 and not test&train
 prior=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent in peers:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):prior.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for pattern in ('*/games.json','*/collection-games.json'):
  for p in Path('runs').glob(pattern):
   if p.parent not in peers:prior.update(r['layout_hash'] for r in json.loads(p.read_text()))
 assert not (test|train)&prior
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
 brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False);brain.eval()
 a,l,c,y=[v[:64] for v in torch.load(CACHE/'train.pt',weights_only=False)];counts={};cache_max_error=0.
 for seed in seeds:
  out=target(seed);assert (out/'completed.json').exists()
  assert json.loads((out/'games.json').read_text())==games and json.loads((out/'collection-games.json').read_text())==collection
  old=torch.load(origin(seed)/'best-dagger.pt',weights_only=False);oldrows=pickle.loads((origin(seed)/'dataset.pkl').read_bytes())['train'];rows=pickle.loads((out/'dataset.pkl').read_bytes())['train']
  same_rows(rows[:len(oldrows)],oldrows)
  newrows=[]
  for rnd in range(1,4):
   rr=pickle.loads((out/f'round-{rnd}-positions.pkl').read_bytes());newrows.extend(rr)
   allowed={r['layout_hash'] for r in collection[(rnd-1)*600:rnd*600]};assert all(r['layout_hash'] in allowed for r in rr)
   cached=torch.load(out/f'round-{rnd}-activity.pt',weights_only=False);assert len(cached[0])==len(rr)
   indices=np.linspace(0,len(rr)-1,4,dtype=int);x,legal,ctx,labels=arrays([rr[i] for i in indices]);fresh=activity_map(brain,x,ctx)
   error=float((fresh-cached[0][indices]).abs().max());cache_max_error=max(cache_max_error,error);torch.testing.assert_close(fresh,cached[0][indices],atol=1e-6,rtol=1e-5)
   for expected,stored in zip((legal,ctx,labels),cached[1:]):torch.testing.assert_close(expected,stored[indices],atol=0,rtol=0)
  same_rows(newrows,rows[len(oldrows):]);counts[seed]=len(newrows)
  for r in newrows:
   b=r['board'].reshape(16,16)[:7,:7].flatten();legal=b==-1;info=analyze(np.eye(10,dtype=np.float32)[b+1].flatten(),legal,7,7);expected=np.zeros(256,bool)
   for i in info.safe:expected[i//7*16+i%7]=True
   assert expected.any();np.testing.assert_array_equal(expected,r['labels'])
   native=r['learner_action']//16*7+r['learner_action']%16;assert legal[native] and r['learner_safe']==bool(native in info.safe)
  for name in ('control','expanded'):
   for kind in ('best','latest'):
    state=torch.load(out/f'{kind}-{name}.pt',weights_only=False);assert state['parent_step']==old['step'];hs=[]
    assert all(torch.isfinite(v).all() for v in state['head'].values())
    for key,os in state['optimizer']['state'].items():
     assert float(os['step'])==float(old['optimizer']['state'][key]['step'])+state['step']
     assert all(torch.isfinite(v).all() for v in os.values() if torch.is_tensor(v))
    for _ in range(2):
     h=ActivityHead(True);h.load_state_dict(state['head']);opt=torch.optim.Adam(h.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(state['optimizer']))
     rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng']);ix=rng.integers(64,size=64)
     opt.zero_grad();loss=equivalent_loss(h(a[ix],c[ix]),l[ix],y[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True);opt.step();hs.append(h)
    for k,v in hs[0].state_dict().items():torch.testing.assert_close(v,hs[1].state_dict()[k],atol=0,rtol=0)
 assert all(digest(p)==sha for p,sha in json.loads((OUT/'manifest.json').read_text())['hashes'].items())
 name='independent_verification.json' if len(seeds)==3 else f'independent_verification-{seeds[0]}.json'
 write_json(OUT/name,dict(new_public_labels_verified=counts,prior_overlap=0,original_hashes_unchanged=True,optimizer_steps_preserved=True,checkpoint_replays_exact=4*len(seeds),cache_sample_max_error=cache_max_error));op.close();print('PASS',counts,cache_max_error)
if __name__=='__main__':main()
