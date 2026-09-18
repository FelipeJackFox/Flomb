"""Learned fusion of four aligned orientations versus the mean of inverse-rotated logits."""
import copy,hashlib,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import arrays,write_json
from experiments.rotation_training_data import rotation_features
from experiments.rotation_readout import RotationPolicy
from experiments.orientation_fusion import FusionHead,FusionPolicy,aligned_stack
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games

OUT=Path('runs/orientation-fusion-001');PARENT=Path('runs/early-interface-001/latest-joint.pt');SEED=20261110
ARMS=('mean','fusion','fusion_aug');PARAMETERS=dict(mean=12769,fusion=12739,fusion_aug=12739)

def new_head(arm):return ActivityHead(True) if arm=='mean' else FusionHead()

def main(out=OUT,seed=SEED):
    global OUT,SEED
    OUT=Path(out);SEED=seed
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl'),Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/nine-dagger-001/dataset.pkl'),Path('runs/deep-coverage-001/states-autonomous.pkl')]
    hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(seed=SEED,updates=3000,batch=64,hashes=hashes,parameters=PARAMETERS,initialization='fresh heads and Adam, same seed; fusion arms share initialization',primary='9 fusion_aug minus mean, four brain forwards per decision in every arm',secondary=['9 fusion minus mean','7'],checkpoint='fixed last3000',scope='frozen encoder previously adapted with1 cycle',cycles=1))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/report_orientation_fusion.py'),Path('experiments/orientation_fusion.py'),Path('experiments/spatial_decoder.py'),Path('experiments/rotation_training_data.py'),Path('experiments/rotation_readout.py')]:shutil.copy2(p,OUT/'source'/p.name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    data=pickle.loads(paths[3].read_bytes());sets=[data['train'],pickle.loads(paths[5].read_bytes())['train'],pickle.loads(paths[6].read_bytes())['train'],pickle.loads(paths[7].read_bytes())]
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(torch.load(PARENT,weights_only=False)['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        brain.cycles=1;frozen=tensor_hash(brain.state_dict());cache=[]
        for index,rows in enumerate(sets):
            write_json(OUT/'progress.json',dict(phase='rotation_cache',dataset=index,total=4))
            # (N,4,...): orientation m of row i is A(R^m x_i) with its rotated mask and labels.
            mapped=tuple(v.reshape(len(rows),4,*v.shape[1:]) for v in rotation_features(brain,rows));cache.append(mapped)
            _,l,c,y=arrays(rows)
            for a,b in zip((l,c,y),(v[:,0] for v in mapped[1:])):torch.testing.assert_close(a,b,rtol=0,atol=0)
        write_json(OUT/'input-verification.json',dict(labels_masks_context_exact=True,aligned_rotations=True))
        groups={}
        for i,r in enumerate(sets[0]):groups.setdefault((r['size'],r['category']),[]).append(i)
        groups=[np.array(groups[k]) for k in sorted(groups)];pairs={}
        for arm in ARMS:
            torch.manual_seed(SEED);head=new_head(arm);assert sum(p.numel() for p in head.parameters())==PARAMETERS[arm]
            init=tensor_hash(head.state_dict());opt=torch.optim.Adam(head.parameters(),lr=.001);rng=np.random.default_rng(SEED);rot_rng=np.random.default_rng(SEED+100);draws=hashlib.sha256()
            for step in range(1,3001):
                indices=[np.array([int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(16)],np.int64)]
                indices += [rng.integers(len(r),size=16,dtype=np.int64) for r in sets[1:]]
                for ix in indices:draws.update(ix.tobytes())
                # Drawn in every arm so the position stream stays identical; only fusion_aug uses it.
                turns=[rot_rng.integers(4,size=len(ix)) for ix in indices]
                if arm!='fusion_aug':turns=[np.zeros_like(t) for t in turns]
                a4,c=[torch.cat([p[ix] for p,ix in zip(parts,indices)]) for parts in (tuple(s[0] for s in cache),tuple(s[2][:,0] for s in cache))]
                j=torch.from_numpy(np.concatenate(turns))
                l,y=[torch.cat([s[n][ix,t] for s,ix,t in zip(cache,indices,turns)]) for n in (1,3)]
                logits=head(a4[:,0],c) if arm=='mean' else head(aligned_stack(a4,c,j),c)
                opt.zero_grad();loss=equivalent_loss(logits,l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%250==0:
                    state=dict(head=copy.deepcopy(head.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,arm=arm)
                    torch.save(state,OUT/f'latest-{arm}.pt');rec=dict(phase='training',arm=arm,step=step,loss=float(loss));write_json(OUT/'progress.json',rec);print(rec,flush=True)
            pairs[arm]=dict(initial=init,draws=draws.hexdigest())
        assert len({p['draws'] for p in pairs.values()})==1 and pairs['fusion']==pairs['fusion_aug'];write_json(OUT/'pairing.json',pairs)
        chosen={a:dict(path=str(OUT/f'latest-{a}.pt'),sha256=digest(OUT/f'latest-{a}.pt')) for a in ARMS}
        write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        del cache
        for arm in ARMS:
            head=new_head(arm);head.load_state_dict(torch.load(OUT/f'latest-{arm}.pt',weights_only=False)['head']);head.requires_grad_(False);rows=[]
            for start in range(0,750,50):
                memo=ActivityMemo(brain)
                policy=RotationPolicy(MemoPolicy(memo,head)) if arm=='mean' else FusionPolicy(memo,head)
                rows.extend(evaluate_games(policy,games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items())
        assert digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,identical_samples=True,fusion_arms_identical_initialization=True,endpoint_fixed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
