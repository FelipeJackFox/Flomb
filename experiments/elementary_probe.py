"""Frozen-checkpoint probes of one-clue inference on controlled partial reveals.

Not reachable-game win rate: zero flood opening is deliberately disabled.
Labels concern the designated clue alone; other clues may resolve ambiguity.
Split entire hidden-neighbor masks before constructing boards. Same 5x5/3 mines
as the existing model, no flags, no hidden truth in neural features.
"""
import hashlib,itertools,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.backbone import encode_visible
from experiments.train import public_context

OUT=Path('runs/elementary-probe-001')
NEIGH=[(r,c) for r in (-1,0,1) for c in (-1,0,1) if (r,c)!=(0,0)]

def dataset():
 rng=np.random.default_rng(20261001)
 masks=[tuple(c) for h in (2,3) for c in itertools.combinations(range(8),h)]
 rng.shuffle(masks);groups=dict(train=masks[:50],valid=masks[50:67],test=masks[67:])
 data={};seen=set()
 for split,count in [('train',600),('valid',240),('test',240)]:
  rows=[]
  while len(rows)<count:
   mask=groups[split][int(rng.integers(len(groups[split])))];h=len(mask);label=len(rows)%3
   k=[0,h,1][label];cr,cc=rng.integers(1,4,size=2)
   hidden=[(cr+NEIGH[i][0],cc+NEIGH[i][1]) for i in mask]
   truth=np.zeros((5,5),bool)
   if k:
    for j in rng.choice(h,k,replace=False):truth[hidden[j]]=True
   outside=[(r,c) for r in range(5) for c in range(5) if abs(r-cr)>1 or abs(c-cc)>1]
   for j in rng.choice(len(outside),3-k,replace=False):truth[outside[j]]=True
   board=np.full((16,16),-2,np.int8);board[:5,:5]=-1
   for r in range(cr-1,cr+2):
    for c in range(cc-1,cc+2):
     if (r,c) not in hidden:board[r,c]=truth[max(0,r-1):min(5,r+2),max(0,c-1):min(5,c+2)].sum()
   assert board[cr,cc]==k and truth.sum()==3
   sig=board.tobytes()+bytes([cr,cc])
   if sig in seen:continue
   seen.add(sig);target=hidden[int(rng.integers(h))]
   rows.append(dict(board=board.reshape(-1),center=(int(cr),int(cc)),target=target,label=label,mask=mask))
  data[split]=rows
 for a,b in [('train','test'),('train','valid'),('valid','test')]:assert not set(groups[a])&set(groups[b])
 return data,groups

@torch.no_grad()
def extract(model,rows):
 result={k:[] for k in ['raw','encoder','sensory','cycle1','cycle2','cycle3','decoder_hidden','logit']}
 for offset in range(0,len(rows),16):
  batch=rows[offset:offset+16];x=torch.from_numpy(encode_visible([r['board'] for r in batch])[0]);b=len(batch)
  context=torch.tensor(np.array([public_context(5,3)]*b));grid=x.reshape(b,16,16,10).permute(0,3,1,2)
  encoded=model.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],1))*(grid.sum(1,keepdim=True)!=0)
  coords=[[ (rr,cc) for rr in range(r['center'][0]-1,r['center'][0]+2) for cc in range(r['center'][1]-1,r['center'][1]+2)] for r in batch]
  result['raw'].extend([grid[i,:,tuple(zip(*p))[0],tuple(zip(*p))[1]].flatten().numpy() for i,p in enumerate(coords)])
  result['encoder'].extend([encoded[i,:,tuple(zip(*p))[0],tuple(zip(*p))[1]].flatten().numpy() for i,p in enumerate(coords)])
  sg=2*model.input_xy[None]*4/15-1
  samples=F.grid_sample(encoded,sg.expand(b,-1,-1)[:,:,None],align_corners=True).squeeze(-1)
  sensory=samples[:,model.input_channel,torch.arange(len(model.input_channel))]
  # Actual injected neurons nearest to each of the nine designated patch cells,
  # one per L1/L2/L3 type; preserves a common local sampling scheme.
  for i,p in enumerate(coords):
   inds=[]
   for rr,cc in p:
    dist=((model.input_xy-torch.tensor([cc/4,rr/4]))**2).sum(1)
    for ch in range(3):inds.append(int(dist.masked_fill(model.input_channel!=ch,float('inf')).argmin()))
   result['sensory'].append(sensory[i,inds].numpy())
  drive=x.new_zeros(model.operator.shape[0],b).index_copy(0,model.input_index,sensory.T)
  state=torch.tanh(drive);values=model.base*(model.edge_log_gain.clamp(-2,2).exp() if model.plastic else 1)
  gain=model.node_log_gain.clamp(-1,1).exp()[:,None]
  model.operator.bind(values.numpy(),force=True)
  for cycle in range(1,4):
   state=torch.tanh(torch.from_numpy(model.operator.multiply(values.numpy(),(gain*state).numpy()))+.15*drive)
   pooled=(state[model.out_5]*model.weight_5[:,:,:,None]).sum(2).permute(2,0,1)
   for i,p in enumerate(coords):result[f'cycle{cycle}'].append(pooled[i,[rr*5+cc for rr,cc in p]].flatten().numpy())
  # Original head applied at target. Probes cannot turn a scalar logit into
  # an autonomous-game claim; they only measure its association with the label.
  target=torch.stack([pooled[i,r['target'][0]*5+r['target'][1]] for i,r in enumerate(batch)])
  dh=model.decoder[1](model.decoder[0](torch.cat([target*10,context],1)))
  result['decoder_hidden'].extend(dh.numpy());result['logit'].extend(model.decoder[2](dh).numpy())
 return {k:np.array(v) for k,v in result.items()}


def probe(features,data):
 out={}
 y={s:torch.tensor([r['label'] for r in data[s]]) for s in data}
 for stage in features['train']:
  tr=features['train'][stage];mean=tr.mean(0);sd=np.maximum(tr.std(0),1e-5)
  x={s:torch.tensor((features[s][stage]-mean)/sd) for s in data};runs=[]
  for seed in (0,1,2):
   torch.manual_seed(seed);m=nn.Sequential(nn.Linear(tr.shape[1],64),nn.Tanh(),nn.Linear(64,3))
   optim=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);best=float('inf');selected=None;stepbest=0
   for step in range(1,601):
    optim.zero_grad();loss=F.cross_entropy(m(x['train']),y['train']);loss.backward();optim.step()
    if step%20==0:
     with torch.no_grad():vl=float(F.cross_entropy(m(x['valid']),y['valid']))
     if vl<best:best=vl;selected={k:v.detach().clone() for k,v in m.state_dict().items()};stepbest=step
   m.load_state_dict(selected)
   with torch.no_grad():
    pred=m(x['test']).argmax(1);correct=pred==y['test'];trainacc=float((m(x['train']).argmax(1)==y['train']).float().mean())
   runs.append(dict(seed=seed,selected_step=stepbest,train_accuracy=trainacc,test_correct=int(correct.sum()),n=len(pred),by_class=[int(correct[y['test']==i].sum()) for i in range(3)]))
  out[stage]=runs;print(stage,runs,flush=True)
 return out


def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 if (OUT/'completed.json').exists():raise SystemExit('Preserve completed run')
 data,groups=dataset();(OUT/'dataset.pkl').write_bytes(pickle.dumps(data))
 config=dict(labels=['safe_from_clue','mine_from_clue','undetermined_from_clue'],mask_splits=groups,counts={s:len(r) for s,r in data.items()},limitations=['Synthetic partial reveals; automatic zero flood disabled','Frozen pre-existing checkpoints, probes trained separately','Different feature widths and locations; weak decoding is not proof of information absence','Mask holdout is not rotation-invariant; no claim of unseen symmetry families'])
 (OUT/'config.json').write_text(json.dumps(config,indent=2));mapping=pickle.loads(Path('runs/structural-learning-001/mapping.pkl').read_bytes());allresults={}
 for name in ['retina_plastic','rewired_plastic']:
  path=Path(f'runs/structural-learning-001/training/{name}-20260926.pt')
  checkpoint=torch.load(path,weights_only=False)
  gp='runs/structural-learning-001/rewired.npz' if name.startswith('rewired') else 'data/processed/graph.npz'
  op=PlasticOperator(sparse.load_npz(gp),workers=4);model=RetinaPolicy(op,mapping,plastic=True);model.load_state_dict(checkpoint['model']);model.eval();del checkpoint
  fs={}
  for split,rows in data.items():
   fs[split]=extract(model,rows);print(name,split,'features ready',flush=True)
   (OUT/'progress.json').write_text(json.dumps(dict(model=name,split=split,phase='extraction')))
  np.savez_compressed(OUT/f'{name}-features.npz',**{s+'_'+k:v for s,parts in fs.items() for k,v in parts.items()})
  allresults[name]=probe(fs,data);(OUT/'results.json').write_text(json.dumps(allresults,indent=2));op.close();del model,op,fs
 (OUT/'completed.json').write_text(json.dumps(dict(completed=True)))

if __name__=='__main__':main()
