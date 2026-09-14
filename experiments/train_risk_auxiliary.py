"""Auxiliary certified-risk supervision versus policy-only continuation."""
import copy,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.risk_auxiliary import mine_labels,outputs,certified_loss
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.repeat_adapted_dagger import origin
from experiments.spatial_decoder import ActivityHead
from experiments.compare_input_representation import reserve
from experiments.extend_spatial_decoder import validate
from experiments.capacity_probe import write_json
from experiments.expressive_models import equivalent_loss
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games
OUT=Path('runs/risk-auxiliary-001');SEED=20261002;PARENT=Path(f'runs/wide-reader-001/best-{SEED}-local.pt')

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 paths=[PARENT,BASE/'dataset.pkl',BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl'),origin(SEED)/'dataset.pkl',origin(SEED)/'train.pt',origin(SEED)/'holdout.pt',Path(f'runs/wide-reader-001/experience-{SEED}.pt')]
 hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(seed=SEED,updates=1500,batch=64,aux_weight=.2,head_lr=.001,aux_lr=.001,loss='policy CE +0.2 balanced BCE over publicly certified safe/mine; unknown masked',mixture='32 original +32 inherited DAgger, identical draws',selection='minimum policy holdout loss at0 and every250, same for both',hashes=hashes))
 src=OUT/'source';src.mkdir()
 for p in [Path(__file__),Path('experiments/risk_auxiliary.py'),Path('experiments/test_risk_auxiliary.py'),Path('experiments/diagnose_mine_choices.py'),Path('solver.py')]:shutil.copy2(p,src/p.name)
 games=reserve();write_json(OUT/'games.json',games)
 with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());extra=pickle.loads((origin(SEED)/'dataset.pkl').read_bytes())['train'];cache=torch.load(origin(SEED)/'train.pt',weights_only=False);valid=torch.load(origin(SEED)/'holdout.pt',weights_only=False);experience=torch.load(paths[-1],weights_only=False)
 write_json(OUT/'progress.json',dict(phase='certify_public_mines'))
 mine=mine_labels(data['train']);newmine=mine_labels(extra);valmine=mine_labels(data['holdout'])
 for labels,part in [(mine,cache),(newmine,experience),(valmine,valid)]:assert not (labels&part[3]).any() and not (labels&~part[1]).any()
 torch.save(dict(original=mine,experience=newmine,holdout=valmine),OUT/'mine-labels.pt')
 groups={}
 for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
 groups=[np.array(groups[k]) for k in sorted(groups)];parent=torch.load(PARENT,weights_only=False);selected={};pair={}
 for arm in ('control','auxiliary'):
  head=ActivityHead(True);head.load_state_dict(parent['head']);opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(parent['optimizer']));aux=torch.nn.Conv2d(32,1,1);torch.nn.init.zeros_(aux.weight);torch.nn.init.zeros_(aux.bias);auxopt=torch.optim.Adam(aux.parameters(),lr=.001)
  rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['rng']);draws=hashlib.sha256();best=float('inf');history=[]
  for step in range(1501):
   if step:
    ix=np.array([int(rng.choice(groups[((parent['step']+step)*64+j)%len(groups)])) for j in range(32)],np.int64);jx=rng.integers(len(extra),size=32,dtype=np.int64);draws.update(ix.tobytes()+jx.tobytes())
    a,l,c,y=[torch.cat([u[ix],v[jx]]) for u,v in zip(cache,experience)];mm=torch.cat([mine[ix],newmine[jx]]);opt.zero_grad();auxopt.zero_grad()
    if arm=='auxiliary':z,risk=outputs(head,aux,a,c);loss=equivalent_loss(z,l,y)+.2*certified_loss(risk,y,mm)
    else:loss=equivalent_loss(head(a,c),l,y)
    loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
    if arm=='auxiliary':torch.nn.utils.clip_grad_norm_(aux.parameters(),5.,error_if_nonfinite=True);auxopt.step()
   if step%250==0:
    value=validate(head,valid)
    with torch.no_grad():
     a,l,c,y=valid;_,risk=outputs(head,aux,a,c);auxloss=float(certified_loss(risk,y,valmine));known=y|valmine;correct=int(((risk>0)==valmine)[known].sum());n=int(known.sum())
    state=dict(head=copy.deepcopy(head.state_dict()),encoder=parent['encoder'],optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),aux=copy.deepcopy(aux.state_dict()),aux_optimizer=copy.deepcopy(auxopt.state_dict()),step=step,validation_loss=value)
    if value<best:best=value;chosen=step;torch.save(state,OUT/f'best-{arm}.pt')
    torch.save(state,OUT/f'latest-{arm}.pt');record=dict(phase='training',arm=arm,step=step,validation_loss=value,auxiliary_holdout_loss=auxloss,certified_correct=correct,certified_n=n,chosen=chosen);history.append(record);write_json(OUT/'progress.json',record);write_json(OUT/f'history-{arm}.json',history);print(record,flush=True)
  selected[arm]=dict(step=chosen,validation_loss=best,sha256=digest(OUT/f'best-{arm}.pt'));pair[arm]=draws.hexdigest()
 assert pair['control']==pair['auxiliary'];write_json(OUT/'pairing.json',pair);write_json(OUT/'selection.json',selected);seal=digest(OUT/'selection.json');write_json(OUT/'selection-seal.json',dict(sha256=seal))
 op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4);brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True);brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.encoder.load_state_dict(parent['encoder']);brain.requires_grad_(False);frozen=tensor_hash(brain.state_dict());memo=ActivityMemo(brain);results={}
 for arm in ('baseline','control','auxiliary'):
  saved=parent if arm=='baseline' else torch.load(OUT/f'best-{arm}.pt',weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head']);write_json(OUT/'progress.json',dict(phase='evaluation',arm=arm,games=500));rows=evaluate_games(MemoPolicy(memo,head),games);write_json(OUT/f'{arm}-games.json',rows);played=[r for r in rows if not r['automatic']]
  results[arm]=dict(wins=sum(r['won'] for r in played),n=len(played),automatic_excluded=len(rows)-len(played),known_mine_choices=sum(r['known_mine_choices'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played));write_json(OUT/'results.json',results);print(arm,results[arm],flush=True)
 assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items()) and digest(OUT/'selection.json')==seal
 write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,paired_draws=True,selection_sealed=True,auxiliary_not_used_for_actions=True,certificates_disjoint_and_legal=True));write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));op.close()
if __name__=='__main__':main()
