import copy,json,pickle
from pathlib import Path
import numpy as np
import torch
from solver import analyze
from experiments.prioritize_mine_errors import OUT,target,parent,origin,SEEDS,CACHE,digest,mistakes
from experiments.spatial_decoder import ActivityHead
from experiments.expressive_models import equivalent_loss
from experiments.capacity_probe import write_json

def main():
 torch.set_num_threads(1);peers={OUT,*[target(s) for s in SEEDS]};test={r['layout_hash'] for r in json.loads((OUT/'games.json').read_text())};assert len(test)==500;used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent in peers:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for pat in ('*/games.json','*/collection-games.json'):
  for p in Path('runs').glob(pat):
   if p.parent not in peers:used.update(r['layout_hash'] for r in json.loads(p.read_text()))
 assert not test&used
 a,l,c,y=[v[:64] for v in torch.load(CACHE/'train.pt',weights_only=False)];counts={};terminal_fit={}
 for seed in SEEDS:
  out=target(seed);assert (out/'trained.json').exists();rows=pickle.loads((out/'dataset.pkl').read_bytes())['train'];mines=torch.load(out/'mine-labels.pt',weights_only=False);counts[seed]=len(rows)
  for i,r in enumerate(rows):
   n=r['size'];b=r['board'].reshape(16,16)[:n,:n].flatten();info=analyze(np.eye(10,dtype=np.float32)[b+1].flatten(),b==-1,n,r['mines']);expected=torch.zeros(256,dtype=torch.bool)
   assert info.safe
   for k in info.mines:expected[k//n*16+k%n]=True
   torch.testing.assert_close(mines[i],expected,atol=0,rtol=0)
  old=torch.load(parent(seed)/'best-expanded.pt',weights_only=False);cache=torch.load(parent(seed)/'inherited.pt',weights_only=False)
  for rnd in range(1,(old['step']+749)//750+1):part=torch.load(parent(seed)/f'round-{rnd}-activity.pt',weights_only=False);cache=tuple(torch.cat([x,z]) for x,z in zip(cache,part))
  head=ActivityHead(True);head.load_state_dict(old['head']);hard,_=mistakes(head,cache,mines)
  terminal_fit[seed]={}
  for name in ('control','priority'):
   np.testing.assert_array_equal(hard,np.load(out/f'hard-{name}-0.npy'))
   for kind in ('best','latest'):
    st=torch.load(out/f'{kind}-{name}.pt',weights_only=False)
    if kind=='latest':
     probe=ActivityHead(True);probe.load_state_dict(st['head']);errors,safe=mistakes(probe,cache,mines);terminal_fit[seed][name]=dict(known_mine_choices=len(errors),safe_choices=safe,positions=len(rows),validation_loss=st['validation_loss'])
    assert all(torch.isfinite(v).all() for v in st['head'].values());hs=[]
    for k,v in st['optimizer']['state'].items():assert float(v['step'])==float(old['optimizer']['state'][k]['step'])+st['step']
    for _ in range(2):
     h=ActivityHead(True);h.load_state_dict(st['head']);opt=torch.optim.Adam(h.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(st['optimizer']));rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(st['rng']);ix=rng.integers(64,size=64)
     opt.zero_grad();loss=equivalent_loss(h(a[ix],c[ix]),l[ix],y[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True);opt.step();hs.append(h)
    for k,v in hs[0].state_dict().items():torch.testing.assert_close(v,hs[1].state_dict()[k],atol=0,rtol=0)
 assert all(digest(p)==sha for p,sha in json.loads((OUT/'manifest.json').read_text())['hashes'].items())
 write_json(OUT/'terminal-fit.json',terminal_fit)
 write_json(OUT/'independent_verification.json',dict(prior_test_overlap=0,public_mine_label_rows=counts,initial_error_pools_exact=True,checkpoint_replays_exact=12,optimizer_steps_preserved=True,original_hashes_unchanged=True));print('PASS',counts)
if __name__=='__main__':main()
