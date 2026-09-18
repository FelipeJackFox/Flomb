"""Matched loss-only continuation: uniform safe targets versus safe-set mass."""
import copy,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import write_json
from experiments.expand_dagger_experience import features
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.expressive_models import equivalent_loss
from experiments.safe_set_loss import safe_set_loss
from experiments.scaled_train import evaluate_games

OUT=Path('runs/safe-mass-001');PARENT=Path('runs/deep-coverage-001/latest-autonomous.pt')

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl'),Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/nine-dagger-001/dataset.pkl'),Path('runs/deep-coverage-001/states-autonomous.pkl')]
    hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(seed=20261002,updates=1500,batch=64,hashes=hashes,primary='9 safe mass minus uniform CE',secondary='7retention and vsparent',mixture='16original+16old7+16old9+16new autonomous9',checkpoint='fixed last1500'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/report_safe_mass.py'),Path('experiments/safe_set_loss.py'),Path('experiments/test_safe_set_loss.py')]:shutil.copy2(p,OUT/'source'/p.name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    parent=torch.load(PARENT,weights_only=False);data=pickle.loads(paths[3].read_bytes());seven=pickle.loads(paths[5].read_bytes())['train'];nine=pickle.loads(paths[6].read_bytes())['train'];fresh=pickle.loads(paths[7].read_bytes())
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(parent['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        frozen=tensor_hash(brain.state_dict());write_json(OUT/'progress.json',dict(phase='recompute_features'))
        caches=[features(ActivityMemo(brain),r) for r in (data['train'],seven,nine,fresh)]
        groups={}
        for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
        groups=[np.array(groups[k]) for k in sorted(groups)];pairs={}
        for arm in ('uniform','mass'):
            head=ActivityHead(True);head.load_state_dict(parent['head']);opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(parent['optimizer']))
            rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['rng']);draws=hashlib.sha256()
            for step in range(1,1501):
                indices=[np.array([int(rng.choice(groups[((1500+step)*64+j)%len(groups)])) for j in range(16)],np.int64)]
                indices += [rng.integers(n,size=16,dtype=np.int64) for n in (len(seven),len(nine),len(fresh))]
                for ix in indices:draws.update(ix.tobytes())
                a,l,c,y=[torch.cat([part[ix] for part,ix in zip(parts,indices)]) for parts in zip(*caches)]
                opt.zero_grad();z=head(a,c);loss=equivalent_loss(z,l,y) if arm=='uniform' else safe_set_loss(z,l,y)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%250==0:
                    state=dict(head=copy.deepcopy(head.state_dict()),encoder=parent['encoder'],optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step)
                    torch.save(state,OUT/f'latest-{arm}.pt');rec=dict(phase='training',arm=arm,step=step);write_json(OUT/'progress.json',rec);print(rec,flush=True)
            pairs[arm]=draws.hexdigest()
        assert pairs['uniform']==pairs['mass'];write_json(OUT/'pairing.json',pairs)
        candidates={'baseline':PARENT,'uniform':OUT/'latest-uniform.pt','mass':OUT/'latest-mass.pt'}
        chosen={a:dict(path=str(p),sha256=digest(p)) for a,p in candidates.items()};write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        for arm,path in candidates.items():
            head=ActivityHead(True);head.load_state_dict(torch.load(path,weights_only=False)['head']);rows=[]
            for start in range(0,750,50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items())
        assert digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,identical_minibatches=True,endpoint_fixed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
