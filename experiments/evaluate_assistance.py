"""Visible-only intervention diagnostic; assisted wins are NOT autonomous learning."""
import hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import write_json
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games
from experiments.backbone import encode_visible
from experiments.train import visible,public_context
from solver import analyze
from minesweeper import Minesweeper

OUT=Path('runs/assistance-diagnostic-001');PARENT=Path('runs/nine-interface-001/latest-joint.pt')
MODES=('autonomous','safe_half','safe_all','public_solver')

def correction_coin(seed,step):
    return int.from_bytes(hashlib.sha256(f'mosca-correction:{seed}:{step}'.encode()).digest()[:8],'big') < 2**63

def choose(mode,proposal,info,seed,step,legal):
    eligible=bool(info.safe and proposal not in info.safe)
    if mode=='public_solver':
        candidates=sorted(info.safe) if info.safe else [int(i) for i in np.flatnonzero(legal) if int(i) not in info.mines]
        if not candidates:candidates=list(map(int,np.flatnonzero(legal)))
        return min(candidates,key=lambda i:(info.mine_probability[i],i)),eligible,True
    correct=eligible and (mode=='safe_all' or (mode=='safe_half' and correction_coin(seed,step)))
    return min(info.safe) if correct else proposal,eligible,bool(correct)

@torch.no_grad()
def evaluate(model,games,mode):
    assert mode in MODES
    model.eval();result=[];active=[];cursor=0
    while cursor<len(games) or active:
        while cursor<len(games) and len(active)<16:
            g=games[cursor];cursor+=1;env=Minesweeper(g['seed'],g['size'],g['mines'])
            row=dict(**g,automatic=bool(env.won),clicks=0,eligible=0,corrections=0,teacher_actions=0,trace=[])
            if env.done:result.append(dict(**row,won=bool(env.won),death_with_safe_available=False))
            else:active.append((env,row))
        if not active:continue
        x,legal=encode_visible([visible(e) for e,r in active]);c=np.array([public_context(e.size,e.mine_count) for e,r in active])
        z=model(torch.from_numpy(x),torch.from_numpy(c));proposals=z.masked_fill(~torch.from_numpy(legal),-1e9).argmax(1).tolist();remaining=[]
        for (env,row),proposal in zip(active,proposals):
            native=(proposal//16)*env.size+proposal%16
            info=analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
            action,eligible,delegated=choose(mode,native,info,row['seed'],row['clicks'],env.legal_mask())
            row['trace'].append(dict(visible=env.visible.tolist(),proposal=native,action=action,eligible=eligible,delegated=delegated,method=info.probability_method))
            row['eligible']+=int(eligible);row['corrections']+=int(action!=native);row['teacher_actions']+=int(delegated)
            env.step(action);row['clicks']+=1
            if env.done:result.append(dict(**row,won=bool(env.won),death_with_safe_available=bool(not env.won and info.safe)))
            else:remaining.append((env,row))
        active=remaining
    return sorted(result,key=lambda r:(r['size'],r['seed']))

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
    hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(checkpoint_fixed=True,training=False,hashes=hashes,modes=MODES,scope='intervention diagnostic; only autonomous arm is autonomous',games=500))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/report_assistance.py'),Path('solver.py'),Path('experiments/scaled_train.py')]:shutil.copy2(p,OUT/'source'/p.name)
    games=[g for g in reserve(OUT)['games'] if g['size']==9];assert len(games)==500
    write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);saved=torch.load(PARENT,weights_only=False);brain.encoder.load_state_dict(saved['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        head=ActivityHead(True);head.load_state_dict(saved['head']);frozen=tensor_hash(brain.state_dict())
        model=MemoPolicy(ActivityMemo(brain),head)
        ref=evaluate_games(model,games[:16]);actual=evaluate(model,games[:16],'autonomous')
        for a,b in zip(ref,actual):
            for k in ('seed','size','layout_hash','won','automatic','clicks','death_with_safe_available'):assert a[k]==b[k]
        write_json(OUT/'preflight.json',dict(autonomous_matches_existing_evaluator=True,games=16))
        for mode in MODES:
            rows=[]
            for start in range(0,500,50):
                rows.extend(evaluate(MemoPolicy(ActivityMemo(brain),head),games[start:start+50],mode));write_json(OUT/f'{mode}-games.json',rows)
                progress=dict(phase='evaluation',mode=mode,completed=len(rows),total=500);write_json(OUT/'progress.json',progress);print(progress,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items())
        write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,no_training=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
