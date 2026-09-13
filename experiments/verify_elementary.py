"""Verify labels/splits and extraction parity with the unmodified forward pass."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.elementary_probe import OUT,NEIGH
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.backbone import encode_visible
from experiments.train import public_context

def main():
 torch.set_num_threads(1);data=pickle.loads((OUT/'dataset.pkl').read_bytes());checks={}
 for split,rows in data.items():
  for r in rows:
   board=r['board'].reshape(16,16);cr,cc=r['center'];neighbors=[(cr+dr,cc+dc) for dr,dc in NEIGH];h=sum(board[p]==-1 for p in neighbors);k=int(board[cr,cc])
   # Independent enumeration of assignments allowed by this clue: target has
   # probability k/h; classification concerns certainty under this clue only.
   expected=0 if k==0 else 1 if k==h else 2
   assert r['label']==expected and board[r['target']]==-1 and h in (2,3)
 for a,b in [('train','valid'),('train','test'),('valid','test')]:
  assert not {r['mask'] for r in data[a]} & {r['mask'] for r in data[b]}
  assert not {r['board'].tobytes() for r in data[a]} & {r['board'].tobytes() for r in data[b]}
 checks['labels_and_split_overlap']='pass'
 mapping=pickle.loads(Path('runs/structural-learning-001/mapping.pkl').read_bytes())
 for name in ['retina_plastic','rewired_plastic']:
  gp='runs/structural-learning-001/rewired.npz' if name.startswith('rewired') else 'data/processed/graph.npz'
  op=PlasticOperator(sparse.load_npz(gp),workers=4);m=RetinaPolicy(op,mapping,True)
  p=Path(f'runs/structural-learning-001/training/{name}-20260926.pt');m.load_state_dict(torch.load(p,weights_only=False)['model'])
  rows=data['test'][:16];x=torch.from_numpy(encode_visible([r['board'] for r in rows])[0]);ctx=torch.tensor(np.array([public_context(5,3)]*len(rows)))
  with torch.no_grad():logits=m(x,ctx).numpy()
  expected=np.array([logits[i,r['target'][0]*16+r['target'][1]] for i,r in enumerate(rows)])
  actual=np.load(OUT/f'{name}-features.npz')['test_logit'][:len(rows),0]
  np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
  checks[name]=dict(forward_max_error=float(np.abs(actual-expected).max()),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest());op.close()
 protected=hashlib.sha256(Path('runs/hybrid-001/checkpoint.pkl').read_bytes()).hexdigest()
 assert protected=='81b09d60395123c0591d8837d2a4419825af51c4a93acb0dbebee31702095ea5'
 checks['protected_hybrid_sha256']=protected
 (OUT/'verification.json').write_text(json.dumps(checks,indent=2));print(checks)
if __name__=='__main__':main()
