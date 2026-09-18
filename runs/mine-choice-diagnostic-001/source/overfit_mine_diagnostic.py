"""Tiny temporary-head fit: can the existing activity/head learn 32 easy mistakes?"""
import json,pickle,copy
from pathlib import Path
import numpy as np
import torch
from experiments.diagnose_mine_choices import OUT,info_from,proof
from experiments.expand_dagger_experience import target,origin,digest
from experiments.spatial_decoder import ActivityHead
from experiments.expressive_models import equivalent_loss
from experiments.capacity_probe import write_json

def main():
 torch.set_num_threads(1);seed=20261002;path=target(seed)/'best-expanded.pt';sha=digest(path);saved=torch.load(path,weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head'])
 rows=pickle.loads((origin(seed)/'dataset.pkl').read_bytes())['train'];cache=torch.load(target(seed)/'inherited.pt',weights_only=False);chosen=[]
 with torch.no_grad():
  for start in range(0,len(rows),128):
   a,l,c,y=[v[start:start+128] for v in cache];acts=head(a,c).masked_fill(~l,-1e9).argmax(1).tolist()
   for j,act in enumerate(acts):
    if y[j,act]:continue
    r=rows[start+j];info=info_from(r);native=act//16*r['size']+act%16
    if proof(r['board'],r['size'],r['mines'],native,info)[0]=='single_clue':chosen.append(start+j)
    if len(chosen)==32:break
   if len(chosen)==32:break
 assert len(chosen)==32
 a,l,c,y=[v[chosen] for v in cache];opt=torch.optim.Adam(head.parameters(),lr=.001);history=[]
 for step in range(501):
  z=head(a,c)
  if step%50==0:
   acts=z.detach().masked_fill(~l,-1e9).argmax(1);hits=int(y[torch.arange(len(y)),acts].sum());history.append(dict(step=step,safe_choices=hits,n=32,loss=float(equivalent_loss(z,l,y).detach())))
  if step==500:break
  opt.zero_grad();loss=equivalent_loss(z,l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
 assert digest(path)==sha
 write_json(OUT/'tiny-overfit.json',dict(scope='training-set diagnostic only, not generalization or game evaluation',seed=seed,indices=chosen,source=str(origin(seed)/'dataset.pkl'),checkpoint_sha256=sha,history=history,temporary_head_only=True,original_unchanged=True));print(json.dumps(history))
if __name__=='__main__':main()
