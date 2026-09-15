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

OUT=Path('runs/adapted-time-001');PARENT=Path('runs/early-interface-001/latest-joint.pt');SEED=20261106

def main(out=OUT,seed=SEED):
    global OUT,SEED
    OUT=Path(out);SEED=seed
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl'),Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/nine-dagger-001/dataset.pkl'),Path('runs/deep-coverage-001/states-autonomous.pkl')]
    hashes={str(p):digest(p) for p in paths};write_json(OUT/'manifest.json',dict(seed=SEED,updates=3000,batch=64,hashes=hashes,parameters=12769,initialization='identical fresh heads and Adam',primary='9 early1 minus late2',secondary='7',checkpoint='fixed last3000',scope='frozen encoder previously adapted with1 cycle; same head budget',cycles=dict(early=1,late=2)))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/report_adapted_time.py'),Path('experiments/spatial_decoder.py'),Path('experiments/compare_input_representation.py')]:shutil.copy2(p,OUT/'source'/p.name)
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
        frozen=tensor_hash(brain.state_dict());cache={'early':[],'late':[]};write_json(OUT/'progress.json',dict(phase='cache_inputs'))
        for rows in sets:
            _,l,c,y=arrays(rows)
            for arm,cycles in [('early',1),('late',2)]:
                brain.cycles=cycles
                mapped=features(ActivityMemo(brain),rows);cache[arm].append(mapped)
                for a,b in zip((l,c,y),mapped[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        early=cache['early'][0][0];late=cache['late'][0][0]
        assert torch.count_nonzero(early)>0 and torch.count_nonzero(late)>0 and not torch.equal(early,late)
        write_json(OUT/'input-verification.json',dict(labels_masks_context_exact=True,one_and_two_nonzero_distinct=True,brain_encoder_fixed=True))
        groups={}
        for i,r in enumerate(sets[0]):groups.setdefault((r['size'],r['category']),[]).append(i)
        groups=[np.array(groups[k]) for k in sorted(groups)];pairs={}
        for arm in ('early','late'):
            torch.manual_seed(SEED);head=ActivityHead(True);assert sum(p.numel() for p in head.parameters())==12769
            init=tensor_hash(head.state_dict());opt=torch.optim.Adam(head.parameters(),lr=.001);rng=np.random.default_rng(SEED);draws=hashlib.sha256()
            for step in range(1,3001):
                indices=[np.array([int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(16)],np.int64)]
                indices += [rng.integers(len(r),size=16,dtype=np.int64) for r in sets[1:]]
                for ix in indices:draws.update(ix.tobytes())
                a,l,c,y=[torch.cat([p[ix] for p,ix in zip(parts,indices)]) for parts in zip(*cache[arm])]
                opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%250==0:
                    state=dict(head=copy.deepcopy(head.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,arm=arm)
                    torch.save(state,OUT/f'latest-{arm}.pt');rec=dict(phase='training',arm=arm,step=step);write_json(OUT/'progress.json',rec);print(rec,flush=True)
            pairs[arm]=dict(initial=init,draws=draws.hexdigest())
        assert pairs['early']==pairs['late'];write_json(OUT/'pairing.json',pairs)
        chosen={a:dict(path=str(OUT/f'latest-{a}.pt'),sha256=digest(OUT/f'latest-{a}.pt')) for a in ('early','late')}
        write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        del cache
        for arm in ('early','late'):
            head=ActivityHead(True);head.load_state_dict(torch.load(OUT/f'latest-{arm}.pt',weights_only=False)['head']);rows=[]
            for start in range(0,750,50):
                brain.cycles=1 if arm=='early' else 2
                policy=MemoPolicy(ActivityMemo(brain),head)
                rows.extend(evaluate_games(policy,games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items())
        assert digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,identical_initial_heads_and_samples=True,endpoint_fixed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
