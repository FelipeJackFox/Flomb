"""Matched fixed-data continuation with/without previously learned7x7 experience."""
import copy,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_risk_auxiliary import BASE,digest,tensor_hash
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import write_json
from experiments.expand_dagger_experience import features
from experiments.extend_spatial_decoder import validate
from experiments.expressive_models import equivalent_loss
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.scaled_train import evaluate_games

OUT=Path('runs/seven-rehearsal-001');PARENT=Path('runs/nine-dagger-001/latest-dagger.pt')

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'dataset.pkl',BASE/'training/retina_plastic-20260926.pt',
        Path('runs/hybrid-001/checkpoint.pkl'),Path('runs/nine-dagger-001/dataset.pkl'),
        Path('runs/nine-dagger-001/train.pt'),Path('runs/nine-dagger-001/holdout.pt'),
        Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/wide-reader-001/experience-20261002.pt')]
    hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(seed=20261002,updates=1500,parent_step=2250,batch=64,hashes=hashes,
        control='32original +32fixed9',replay='16original +16inherited7 +32samefixed9',
        primary='7x7 replay minus control',secondary='9x9 cost and vsparent',checkpoint='fixed last1500 additional'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/report_seven_rehearsal.py'),Path('experiments/spatial_decoder.py'),Path('experiments/scaled_train.py')]:shutil.copy2(p,OUT/'source'/p.name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    parent=torch.load(PARENT,weights_only=False);assert parent['step']==2250
    old=pickle.loads((BASE/'dataset.pkl').read_bytes());nine=pickle.loads(paths[5].read_bytes())['train'];seven=pickle.loads(paths[8].read_bytes())['train']
    cache=torch.load(paths[6],weights_only=False);valid=torch.load(paths[7],weights_only=False);legacy=torch.load(paths[9],weights_only=False)
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[3],weights_only=False)['model']);brain.encoder.load_state_dict(parent['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if hasattr(brain,f'out_{size}'):
                assert torch.equal(getattr(brain,f'out_{size}'),torch.from_numpy(ix)) and torch.equal(getattr(brain,f'weight_{size}'),torch.from_numpy(wt))
            else:
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        frozen=tensor_hash(brain.state_dict());memo=ActivityMemo(brain)
        write_json(OUT/'progress.json',dict(phase='cache_and_parity'))
        cache_errors=[]
        for rows,cached in [(old['train'],cache),(old['holdout'],valid),(seven,legacy)]:
            ix=np.linspace(0,len(rows)-1,8,dtype=int);fresh=features(memo,[rows[i] for i in ix])
            # Different inference batch sizes can change float32 convolution rounding.
            torch.testing.assert_close(fresh[0],cached[0][ix],rtol=1e-5,atol=2e-6)
            cache_errors.append(float((fresh[0]-cached[0][ix]).abs().max()))
            for a,b in zip(fresh[1:],cached[1:]):torch.testing.assert_close(a,b[ix],rtol=0,atol=0)
        source_encoder=torch.load('runs/wide-reader-001/best-20261002-local.pt',weights_only=False)['encoder']
        assert tensor_hash(source_encoder)==tensor_hash(parent['encoder'])
        new=features(memo,nine);torch.save(new,OUT/'nine-features.pt')
        write_json(OUT/'cache-verification.json',dict(labels_context_masks_exact=True,encoder_exact=True,
            activity_rtol=1e-5,activity_atol=2e-6,activity_max_abs_errors=cache_errors,nine_positions=len(nine),seven_positions=len(seven)))
        groups={}
        for i,r in enumerate(old['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
        groups=[np.array(groups[k]) for k in sorted(groups)]
        pairing={}
        for arm in ('control','replay'):
            head=ActivityHead(True);head.load_state_dict(parent['head']);opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(parent['optimizer']))
            rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['rng']);draws=hashlib.sha256();history=[]
            for step in range(1,1501):
                ix=np.array([int(rng.choice(groups[((2250+step)*64+j)%len(groups)])) for j in range(32)],np.int64)
                jx=rng.integers(len(nine),size=32,dtype=np.int64);kx=rng.integers(len(seven),size=16,dtype=np.int64)
                draws.update(ix.tobytes()+jx.tobytes()+kx.tobytes())
                if arm=='control':batch=[torch.cat([a[ix],b[jx]]) for a,b in zip(cache,new)]
                else:batch=[torch.cat([a[ix[:16]],c[kx],b[jx]]) for a,b,c in zip(cache,new,legacy)]
                a,l,c,y=batch;opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%250==0:
                    value=validate(head,valid)
                    state=dict(head=copy.deepcopy(head.state_dict()),encoder=parent['encoder'],optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,parent_step=2250,validation_loss=value)
                    torch.save(state,OUT/f'latest-{arm}.pt');rec=dict(phase='training',arm=arm,step=step,validation_loss=value)
                    history.append(rec);write_json(OUT/f'history-{arm}.json',history);write_json(OUT/'progress.json',rec);print(rec,flush=True)
            pairing[arm]=draws.hexdigest()
        assert pairing['control']==pairing['replay'];write_json(OUT/'pairing.json',pairing)
        candidates={'baseline':PARENT,'control':OUT/'latest-control.pt','replay':OUT/'latest-replay.pt'}
        chosen={a:dict(path=str(p),sha256=digest(p)) for a,p in candidates.items()};write_json(OUT/'evaluation-checkpoints.json',chosen)
        seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        for arm,path in candidates.items():
            saved=torch.load(path,weights_only=False);head=ActivityHead(True);head.load_state_dict(saved['head']);rows=[]
            for start in range(0,750,50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),games[start:start+50]))
                write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items())
        assert digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,paired_indices=True,endpoint_fixed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
