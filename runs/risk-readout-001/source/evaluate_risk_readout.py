"""Predeclared frozen last-checkpoint readout comparison; no additional training."""
import json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments.train_risk_auxiliary import PARENT,BASE,digest,tensor_hash
from experiments.compare_input_representation import reserve
from experiments.capacity_probe import write_json
from experiments.spatial_decoder import ActivityHead
from experiments.risk_auxiliary import outputs
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games
OUT=Path('runs/risk-readout-001');LAST=Path('runs/risk-auxiliary-001/latest-auxiliary.pt')
class RiskReadout(torch.nn.Module):
 def __init__(self,head,aux):super().__init__();self.head,self.aux=head,aux
 def forward(self,a,c):return -outputs(self.head,self.aux,a,c)[1]
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 paths=[PARENT,LAST,BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')];hashes={str(p):digest(p) for p in paths}
 write_json(OUT/'manifest.json',dict(training=False,seed=20261002,checkpoint='fixed last auxiliary1500; no selection on final outcomes',primary='risk readout vs policy readout of identical last checkpoint',secondary='risk readout vs preserved baseline',risk='argmin learned mine logit; not calibrated probability or solver filtering',hashes=hashes));src=OUT/'source';src.mkdir()
 for p in [Path(__file__),Path('experiments/risk_auxiliary.py'),Path('experiments/scaled_train.py')]:shutil.copy2(p,src/p.name)
 games=reserve();write_json(OUT/'games.json',games)
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
 parent=torch.load(PARENT,weights_only=False);last=torch.load(LAST,weights_only=False);assert last['step']==1500 and tensor_hash(last['encoder'])==tensor_hash(parent['encoder'])
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.encoder.load_state_dict(last['encoder']);brain.requires_grad_(False);memo=ActivityMemo(brain);results={}
 for arm in ('baseline','last_policy','last_risk'):
  saved=parent if arm=='baseline' else last;head=ActivityHead(True);head.load_state_dict(saved['head']);aux=torch.nn.Conv2d(32,1,1)
  if arm=='last_risk':aux.load_state_dict(last['aux']);reader=RiskReadout(head,aux)
  else:reader=head
  write_json(OUT/'progress.json',dict(phase='evaluation',arm=arm,games=500));rows=evaluate_games(MemoPolicy(memo,reader),games);write_json(OUT/f'{arm}-games.json',rows);played=[r for r in rows if not r['automatic']]
  results[arm]=dict(wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played),known_mine_choices=sum(r['known_mine_choices'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played));write_json(OUT/'results.json',results);print(arm,results[arm],flush=True)
 assert all(digest(p)==sha for p,sha in hashes.items());write_json(OUT/'verification.json',dict(originals_intact=True,checkpoint_fixed_before_test=True,identical_encoder_for_all=True,no_training=True));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
