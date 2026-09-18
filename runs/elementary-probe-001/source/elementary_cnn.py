"""Task-trained CNN control using exactly the designated 3x3 visible patch."""
import json,pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.elementary_probe import OUT
from experiments.backbone import encode_visible

def main():
 torch.set_num_threads(1);data=pickle.loads((OUT/'dataset.pkl').read_bytes());x={};y={}
 for split,rows in data.items():
  patches=[r['board'].reshape(16,16)[r['center'][0]-1:r['center'][0]+2,r['center'][1]-1:r['center'][1]+2].flatten() for r in rows]
  x[split]=torch.from_numpy(encode_visible(patches)[0]).reshape(-1,3,3,10).permute(0,3,1,2)
  y[split]=torch.tensor([r['label'] for r in rows])
 results=[]
 for seed in (0,1,2):
  torch.manual_seed(seed);m=nn.Sequential(nn.Conv2d(10,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(144,3))
  optim=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);best=float('inf')
  for step in range(1,601):
   optim.zero_grad();loss=F.cross_entropy(m(x['train']),y['train']);loss.backward();optim.step()
   if step%20==0:
    with torch.no_grad():vl=float(F.cross_entropy(m(x['valid']),y['valid']))
    if vl<best:best=vl;state={k:v.detach().clone() for k,v in m.state_dict().items()};chosen=step
  m.load_state_dict(state)
  with torch.no_grad():correct=m(x['test']).argmax(1)==y['test']
  results.append(dict(seed=seed,selected_step=chosen,test_correct=int(correct.sum()),n=len(correct),by_class=[int(correct[y['test']==i].sum()) for i in range(3)]));print(results[-1],flush=True)
 (OUT/'cnn_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
