"""Fixed-budget 9x9 DAgger pilot, retaining small-board data and evaluation."""
import json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments import adapted_dagger
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.train_risk_auxiliary import PARENT,BASE,digest,tensor_hash
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments.scaled_train import evaluate_games
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from minesweeper import Minesweeper

OUT=Path('runs/nine-dagger-001')
MAPPING=Path('runs/difficulty-transfer-002/extended-mapping.pkl')


def reserve(out=None):
    used=prior_layouts(exclude=OUT if out is None else Path(out)); result={'games':[],'collection':[]}
    for split,size,mines,n in [('games',9,12,500),('games',7,7,250),('collection',9,12,900)]:
        found=0
        for seed in range(8300000000+size*100000,8300000000+(size+1)*100000):
            key=identity(Minesweeper(seed,size,mines))
            if key in used:continue
            used.add(key);result[split].append(dict(seed=seed,size=size,mines=mines,layout_hash=key));found+=1
            if found==n:break
        assert found==n
    return result


def main(out=OUT,seed=20261002,splits=None,train_only=False):
    global OUT,PARENT
    OUT=Path(out)
    PARENT=Path(f'runs/wide-reader-001/best-{seed}-local.pt')
    OUT.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),OUT/'driver-source.py')
    shutil.copy2('benchmarks/generalized-collector-verification.json',OUT/'collector-preflight.json')
    splits=reserve() if splits is None else splits
    write_json(OUT/'protocol.json',dict(seed=seed,rounds=3,updates=2250,
        collection='900 new autonomous 9x9/12 games, 300 per round',
        mixture='32 balanced original 5/7 positions +32 aggregated new9; control64 original',
        checkpoint='fixed last2250 for both arms; best old holdout retained only as diagnostic',
        primary='9x9 win rate dagger2250 minus control2250',
        secondary='vs preserved baseline; 7x7 retention and safe-choice ratios',
        evaluation='500 new9 and250 new7, shared across three arms; one seed/brain',
        hashes={str(p):digest(p) for p in [PARENT,MAPPING]}))
    adapted_dagger.main(out=OUT,seed=seed,splits=splits,train_only=True,
        parent_path=PARENT,mapping_path=MAPPING)
    for p in [Path(__file__),Path('experiments/spatial_decoder.py'),Path('experiments/retina_policy.py')]:
        shutil.copy2(p,OUT/'source'/p.name)
    if train_only:return
    paths={'baseline':PARENT,'control':OUT/'latest-control.pt','dagger':OUT/'latest-dagger.pt'}
    fixed={k:dict(path=str(p),sha256=digest(p)) for k,p in paths.items()}
    write_json(OUT/'evaluation-checkpoints.json',fixed)
    seal=digest(OUT/'evaluation-checkpoints.json')
    write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model'])
        parent=torch.load(PARENT,weights_only=False);brain.encoder.load_state_dict(parent['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False)
                brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        frozen=tensor_hash(brain.state_dict())
        for arm,path in paths.items():
            state=torch.load(path,weights_only=False)
            assert tensor_hash(state['encoder'])==tensor_hash(parent['encoder'])
            if arm!='baseline':assert state['step']==2250
            head=ActivityHead(True);head.load_state_dict(state['head']);rows=[]
            for start in range(0,len(splits['games']),50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),splits['games'][start:start+50]))
                write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                progress=dict(phase='evaluation',arm=arm,completed=len(rows),total=len(splits['games']))
                write_json(OUT/'progress.json',progress);print(progress,flush=True)
        assert tensor_hash(brain.state_dict())==frozen
        assert digest(OUT/'evaluation-checkpoints.json')==seal
        assert all(digest(v['path'])==v['sha256'] for v in fixed.values())
        manifest=json.loads((OUT/'manifest.json').read_text())
        assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
        write_json(OUT/'verification.json',dict(frozen_brain=True,originals_intact=True,
            fixed_endpoint2250=True,teacher_actions=0,evaluation_sealed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()


if __name__=='__main__':main()
