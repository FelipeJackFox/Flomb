"""Replicate fixed-budget9x9 adaptation; two new seeds primary."""
import json,pickle,shutil
from pathlib import Path
import torch
from scipy import sparse
from experiments import train_nine_dagger as training
from experiments.train_risk_auxiliary import BASE,digest,tensor_hash
from experiments.capacity_probe import write_json
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games

OUT=Path('runs/nine-dagger-consistency-001')
SEEDS=(20261002,20261003,20261004)
MAPPING=Path('runs/difficulty-transfer-002/extended-mapping.pkl')
def origin(seed):return Path('runs/nine-dagger-001' if seed==SEEDS[0] else f'runs/nine-dagger-{seed}')
def parent(seed):return Path(f'runs/wide-reader-001/best-{seed}-local.pt')


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    protected=[parent(s) for s in SEEDS]+[MAPPING,BASE/'training/retina_plastic-20260926.pt',Path('runs/hybrid-001/checkpoint.pkl')]
    protected += [origin(SEEDS[0])/f'latest-{a}.pt' for a in ('control','dagger')]
    hashes={str(p):digest(p) for p in protected}
    write_json(OUT/'manifest.json',dict(seeds=SEEDS,new_seeds=SEEDS[1:],updates=2250,hashes=hashes,
        primary='mean9x9 dagger minus control on two NEW seeds',secondary='three seeds; vsbaseline and7x7 retention',
        checkpoint='fixed last2250, no game selection',shared_test='500 new9+250 new7',shared_collection=900))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/train_nine_dagger.py'),Path('experiments/adapted_dagger.py'),Path('experiments/report_nine_dagger_consistency.py')]:shutil.copy2(p,OUT/'source'/p.name)
    splits=training.reserve(OUT)
    write_json(OUT/'games.json',splits['games']);write_json(OUT/'collection-games.json',splits['collection'])
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(splits,f)
    for seed in SEEDS[1:]:
        write_json(OUT/'progress.json',dict(phase='training',seed=seed,detail_file=str(origin(seed)/'progress.json')))
        training.main(out=origin(seed),seed=seed,splits=splits,train_only=True)
    chosen={}
    for seed in SEEDS:
        for arm in ('baseline','control','dagger'):
            p=parent(seed) if arm=='baseline' else origin(seed)/f'latest-{arm}.pt'
            state=torch.load(p,weights_only=False)
            if arm!='baseline':assert state['step']==2250
            chosen[f'{seed}-{arm}']=dict(path=str(p),sha256=digest(p))
    write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json')
    write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if hasattr(brain,f'out_{size}'):
                assert torch.equal(getattr(brain,f'out_{size}'),torch.from_numpy(ix)) and torch.equal(getattr(brain,f'weight_{size}'),torch.from_numpy(wt))
            else:
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        for seed in SEEDS:
            original=torch.load(parent(seed),weights_only=False);brain.encoder.load_state_dict(original['encoder']);frozen=tensor_hash(brain.state_dict())
            for arm in ('baseline','control','dagger'):
                key=f'{seed}-{arm}';saved=torch.load(chosen[key]['path'],weights_only=False)
                assert tensor_hash(saved['encoder'])==tensor_hash(original['encoder'])
                head=ActivityHead(True);head.load_state_dict(saved['head']);rows=[]
                for start in range(0,750,50):
                    rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),splits['games'][start:start+50]))
                    write_json(OUT/f'{key}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                    rec=dict(phase='evaluation',seed=seed,arm=arm,completed=len(rows),total=750)
                    write_json(OUT/'progress.json',rec);print(rec,flush=True)
            assert frozen==tensor_hash(brain.state_dict())
        assert all(digest(p)==sha for p,sha in hashes.items())
        assert all(digest(v['path'])==v['sha256'] for v in chosen.values())
        assert digest(OUT/'evaluation-checkpoints.json')==seal
        write_json(OUT/'verification.json',dict(originals_intact=True,encoder_frozen=True,endpoint2250=True,selection_sealed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
