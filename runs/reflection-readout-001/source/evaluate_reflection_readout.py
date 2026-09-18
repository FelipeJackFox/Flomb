"""Same reader and frozen graph, reading after one versus two cycles."""
import copy,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import arrays,write_json
from experiments.expand_dagger_experience import features
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games

from experiments.rotation_readout import RotationPolicy
from experiments.reflection_readout import ReflectionPolicy
OUT=Path('runs/reflection-readout-001');PARENT=Path('runs/early-interface-001/latest-joint.pt')

def main(out=OUT,parent_path=PARENT):
    global OUT,PARENT
    OUT=Path(out);PARENT=Path(parent_path)
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
    hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(hashes=hashes,updates=0,cycles=1,primary='9 eight minus four',aggregation='mean inverse-symmetry logits',extra_compute='eight vs four forwards per decision'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/rotation_readout.py'),Path('experiments/reflection_readout.py'),Path('experiments/report_reflection_readout.py')]:shutil.copy2(p,OUT/'source'/p.name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(torch.load(PARENT,weights_only=False)['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        brain.cycles=1;frozen=tensor_hash(brain.state_dict())
        head=ActivityHead(True);head.load_state_dict(torch.load(PARENT,weights_only=False)['head']);head.requires_grad_(False)
        head_hash=tensor_hash(head.state_dict())
        for arm in ('four','eight'):
            rows=[]
            for start in range(0,750,50):
                policy=MemoPolicy(ActivityMemo(brain),head)
                policy=ReflectionPolicy(policy) if arm=='eight' else RotationPolicy(policy)
                rows.extend(evaluate_games(policy,games[start:start+50]))
                write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and head_hash==tensor_hash(head.state_dict())
        assert all(digest(p)==sha for p,sha in hashes.items())
        write_json(OUT/'verification.json',dict(weights_fixed=True,originals_intact=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
