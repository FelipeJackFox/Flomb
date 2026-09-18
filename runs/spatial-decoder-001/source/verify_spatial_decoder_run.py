"""Independent cache parity for both sizes and activity-zero intervention."""
import hashlib,json,pickle
from pathlib import Path
import torch
from scipy import sparse
from experiments.train_spatial_decoder import OUT,ROOT
from experiments.spatial_decoder import ActivityHead
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.capacity_probe import arrays
from experiments.scaled_train import evaluate_games

class ZeroActivity(torch.nn.Module):
 def __init__(self,head):super().__init__();self.head=head
 def forward(self,x,context):return self.head(x.new_zeros(len(x),10,16,16),context)

def main():
 torch.set_num_threads(1);assert (OUT/'completed.json').exists()
 data=pickle.loads((ROOT/'dataset.pkl').read_bytes());mapping=pickle.loads((ROOT/'mapping.pkl').read_bytes())
 a,legal,ctx,labels=torch.load(OUT/'holdout.pt',weights_only=False)
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,mapping,True)
 brain.load_state_dict(torch.load(ROOT/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False)
 checks={}
 for size in (5,7):
  ix=[i for i,r in enumerate(data['holdout']) if r['size']==size][:16];x,l,c,y=arrays([data['holdout'][i] for i in ix])
  with torch.no_grad():
   expected=brain(x,c);actual=brain.decoder(torch.cat([a[ix].permute(0,2,3,1),c[:,None,None].expand(-1,16,16,-1)],-1)).squeeze(-1).flatten(1)
  torch.testing.assert_close(actual[l],expected[l],rtol=1e-5,atol=1e-5);checks[str(size)]=float((actual[l]-expected[l]).abs().max())
 op.close()
 head=ActivityHead(True);head.load_state_dict(torch.load(OUT/'spatial.pt',weights_only=False)['head'])
 games=json.loads((OUT/'games.json').read_text());rows=evaluate_games(ZeroActivity(head),games)
 (OUT/'spatial-zero-games.json').write_text(json.dumps(rows,indent=2))
 checks['zero_activity_wins']={str(size):dict(wins=sum(r['won'] for r in rows if r['size']==size and not r['automatic']),n=sum(r['size']==size and not r['automatic'] for r in rows)) for size in (5,7)}
 (OUT/'verification.json').write_text(json.dumps(checks,indent=2));print(checks)
if __name__=='__main__':main()
