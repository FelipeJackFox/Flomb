"""Adapt the encoder for1-cycle readout versus fixed-encoder continuation."""
import copy,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash,frozen_hash,validation
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import arrays,write_json
from experiments.joint_interface import JointInterface
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead,activity_map
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.extend_spatial_decoder import validate
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games

OUT=Path('runs/early-interface-001');PARENT=Path('runs/propagation-time-001/latest-early.pt')
ENCODER=Path('runs/deep-coverage-001/latest-autonomous.pt')

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl'),
        Path('runs/nine-dagger-001/dataset.pkl'),Path('runs/adapted-dagger-001/dataset.pkl'),
        Path('runs/deep-coverage-001/states-autonomous.pkl'),ENCODER]
    hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(seed=20261103,updates=750,cycles=1,batch=64,microbatch=16,hashes=hashes,
        primary='9x9 joint minus control',secondary='7retention and vsearly parent',
        mixture='16original+16inherited7+16old9+16fresh9',encoder_lr=.0001,head_lr=.001,clip='5 separately for head and encoder',checkpoint='fixed last750'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/joint_interface.py'),Path('experiments/report_early_interface.py'),Path('experiments/spatial_decoder.py')]:shutil.copy2(p,OUT/'source'/p.name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    parent=torch.load(PARENT,weights_only=False);parent['encoder']=torch.load(ENCODER,weights_only=False)['encoder']
    torch.save(parent,OUT/'baseline.pt')
    data=pickle.loads(paths[3].read_bytes());nine=pickle.loads(paths[5].read_bytes())['train'];seven=pickle.loads(paths[6].read_bytes())['train'];fresh=pickle.loads(paths[7].read_bytes())
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(parent['encoder'])
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if hasattr(brain,f'out_{size}'):
                assert torch.equal(getattr(brain,f'out_{size}'),torch.from_numpy(ix)) and torch.equal(getattr(brain,f'weight_{size}'),torch.from_numpy(wt))
            else:
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        brain.cycles=1
        head=ActivityHead(True);head.load_state_dict(parent['head']);model=JointInterface(brain,head);fixed=frozen_hash(brain)
        write_json(OUT/'progress.json',dict(phase='gradient_preflight'))
        x,l,c,y=arrays([data['train'][0],seven[0],nine[0],fresh[0]])
        torch.testing.assert_close(model.activity(x,c),activity_map(brain,x,c),rtol=1e-5,atol=2e-6)
        params=list(head.parameters())+list(brain.encoder.parameters())
        model.zero_grad();equivalent_loss(model(x,c),l,y).backward();full=[p.grad.clone() for p in params];model.zero_grad()
        for i in (0,2):(equivalent_loss(model(x[i:i+2],c[i:i+2]),l[i:i+2],y[i:i+2])/2).backward()
        for p,g in zip(params,full):torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=2e-6)
        model.zero_grad();equivalent_loss(model(x[2:],c[2:]),l[2:],y[2:]).backward();grad=sum(float(p.grad.abs().sum()) for p in brain.encoder.parameters());assert grad>0
        model.zero_grad()
        from experiments.expand_dagger_experience import features
        write_json(OUT/'progress.json',dict(phase='recompute_one_cycle_features'))
        caches=[features(ActivityMemo(brain),rows) for rows in (data['train'],seven,nine,fresh)]
        valid=features(ActivityMemo(brain),data['holdout'])
        batch_diagnostics=[]
        for rows,cache in zip((data['train'],seven,nine,fresh),caches):
            xx,ll,cc,yy=arrays(rows[:4]);aa=activity_map(brain,xx,cc)
            bx,_,bc,_=arrays(rows[:16])
            replay=ActivityMemo(brain).get(bx,bc)
            # Reproduce cache construction with the same batch and deduplication.
            torch.testing.assert_close(replay,cache[0][:16],rtol=0,atol=0)
            direct16=activity_map(brain,bx,bc)[:4]
            batch_diagnostics.append(dict(batch4_vs16_max=float((aa-direct16).abs().max()),batch4_vs_cache_max=float((aa-cache[0][:4]).abs().max())))
            write_json(OUT/'cache-batch-diagnostics.json',batch_diagnostics)
            torch.testing.assert_close(aa,direct16,rtol=1e-5,atol=1e-5)
            torch.testing.assert_close(aa,cache[0][:4],rtol=1e-5,atol=1e-5)
            for a,b in zip((ll,cc,yy),cache[1:]):torch.testing.assert_close(a,b[:4],rtol=0,atol=0)
        write_json(OUT/'preflight.json',dict(real_forward_parity=True,real_gradient_accumulation_parity=True,nine_encoder_gradient_l1=grad,caches_checked=True))
        groups={}
        for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
        groups=[np.array(groups[k]) for k in sorted(groups)];pairs={}
        for arm in ('control','joint'):
            head.load_state_dict(parent['head']);brain.encoder.load_state_dict(parent['encoder'])
            opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(parent['optimizer']))
            if arm=='joint':opt.add_param_group(dict(params=list(brain.encoder.parameters()),lr=.0001))
            rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['rng']);draws=hashlib.sha256();history=[]
            for step in range(1,751):
                ix=np.array([int(rng.choice(groups[((3000+step)*64+j)%len(groups)])) for j in range(16)],np.int64)
                kx=rng.integers(len(seven),size=16,dtype=np.int64);jx=rng.integers(len(nine),size=16,dtype=np.int64);nx=rng.integers(len(fresh),size=16,dtype=np.int64);draws.update(ix.tobytes()+kx.tobytes()+jx.tobytes()+nx.tobytes())
                model.zero_grad(set_to_none=True)
                if arm=='control':
                    a,l,c,y=[torch.cat([v[ix],w[kx],z[jx],n[nx]]) for v,w,z,n in zip(*caches)];equivalent_loss(head(a,c),l,y).backward()
                else:
                    rows=[data['train'][i] for i in ix]+[seven[i] for i in kx]+[nine[i] for i in jx]+[fresh[i] for i in nx]
                    for i in range(0,64,16):
                        x,l,c,y=arrays(rows[i:i+16]);(equivalent_loss(model(x,c),l,y)/4).backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
                if arm=='joint':torch.nn.utils.clip_grad_norm_(brain.encoder.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                if step%25==0:
                    rec=dict(phase='training',arm=arm,step=step);write_json(OUT/'progress.json',rec);print(rec,flush=True)
                if step%250==0:
                    value=validate(head,valid) if arm=='control' else validation(model,data['holdout'])
                    state=dict(head=copy.deepcopy(head.state_dict()),encoder=copy.deepcopy(brain.encoder.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step,validation_loss=value)
                    torch.save(state,OUT/f'latest-{arm}.pt');history.append(dict(step=step,validation_loss=value));write_json(OUT/f'history-{arm}.json',history)
            pairs[arm]=draws.hexdigest()
        assert pairs['control']==pairs['joint'];write_json(OUT/'pairing.json',pairs)
        candidates={'baseline':OUT/'baseline.pt','control':OUT/'latest-control.pt','joint':OUT/'latest-joint.pt'}
        chosen={a:dict(path=str(p),sha256=digest(p)) for a,p in candidates.items()};write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        for arm,path in candidates.items():
            saved=torch.load(path,weights_only=False);head.load_state_dict(saved['head']);brain.encoder.load_state_dict(saved['encoder']);rows=[]
            for start in range(0,750,50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert fixed==frozen_hash(brain) and all(digest(p)==sha for p,sha in hashes.items())
        assert digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,graph_fixed=True,paired_indices=True,endpoint_fixed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
