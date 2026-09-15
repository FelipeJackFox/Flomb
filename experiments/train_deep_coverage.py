"""Learn from autonomous versus safety-corrected collection; autonomous final test."""
import copy,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.evaluate_assistance import evaluate as collect_trajectories
from experiments.train_joint_interface import BASE,digest,tensor_hash
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import write_json
from experiments.expand_dagger_experience import features
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games
from solver import analyze

OUT=Path('runs/deep-coverage-001');PARENT=Path('runs/nine-interface-001/latest-joint.pt')

def labeled_states(trajectories):
    rows=[]
    for game in trajectories:
        for step,t in enumerate(game['trace']):
            b=np.array(t['visible'],np.int8);info=analyze(np.eye(10,dtype=np.float32)[b+1].flatten(),b==-1,game['size'],game['mines'])
            if not info.safe:continue
            s=game['size'];board=np.full((16,16),-2,np.int8);board[:s,:s]=b.reshape(s,s);labels=np.zeros(256,bool)
            for i in info.safe:labels[(i//s)*16+i%s]=True
            rows.append(dict(seed=game['seed'],size=s,mines=game['mines'],layout_hash=game['layout_hash'],board=board.flatten(),labels=labels,depth=step,hidden=int((b==-1).sum())))
    assert rows
    return rows

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl'),Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/nine-dagger-001/dataset.pkl')]
    hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(seed=20261002,updates=1500,batch=64,hashes=hashes,collection=400,
        mixture='16original+16old7+16old9+16new per-arm',primary='9 autonomous evaluation assisted-trained minus autonomous-trained',
        checkpoint='fixed last1500',teacher='safety corrections only during assisted collection; ZERO at test'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/evaluate_assistance.py'),Path('experiments/report_deep_coverage.py'),Path('solver.py')]:shutil.copy2(p,OUT/'source'/p.name)
    splits=reserve(OUT);games=splits['games'];collection=splits['collection'][:400]
    write_json(OUT/'games.json',games);write_json(OUT/'collection-games.json',collection)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games,collection=collection),f)
    parent=torch.load(PARENT,weights_only=False);data=pickle.loads(paths[3].read_bytes());old7=pickle.loads(paths[5].read_bytes())['train'];old9=pickle.loads(paths[6].read_bytes())['train']
    optstate=copy.deepcopy(parent['optimizer']);optstate['param_groups']=optstate['param_groups'][:1]
    ids=set(optstate['param_groups'][0]['params']);optstate['state']={k:v for k,v in optstate['state'].items() if k in ids}
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=4)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(parent['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        frozen=tensor_hash(brain.state_dict());head=ActivityHead(True);head.load_state_dict(parent['head']);fresh={};stats={}
        for arm,mode in [('autonomous','autonomous'),('assisted','safe_all')]:
            trajectories=[]
            for start in range(0,400,50):
                trajectories.extend(collect_trajectories(MemoPolicy(ActivityMemo(brain),head),collection[start:start+50],mode))
                rec=dict(phase='collection',arm=arm,completed=len(trajectories),total=400);write_json(OUT/'progress.json',rec);print(rec,flush=True)
            write_json(OUT/f'collection-{arm}.json',trajectories);fresh[arm]=labeled_states(trajectories)
            with (OUT/f'states-{arm}.pkl').open('wb') as f:pickle.dump(fresh[arm],f)
            stats[arm]=dict(positions=len(fresh[arm]),mean_depth=float(np.mean([r['depth'] for r in fresh[arm]])),mean_hidden=float(np.mean([r['hidden'] for r in fresh[arm]])),teacher_actions=sum(r['teacher_actions'] for r in trajectories),collection_wins=sum(r['won'] for r in trajectories))
        write_json(OUT/'collection-stats.json',stats)
        write_json(OUT/'progress.json',dict(phase='recompute_adapted_features'))
        def cache(rows):return features(ActivityMemo(brain),rows)
        fixed=[cache(r) for r in (data['train'],old7,old9)];new={a:cache(r) for a,r in fresh.items()}
        for a,v in new.items():torch.save(v,OUT/f'features-{a}.pt')
        groups={}
        for i,r in enumerate(data['train']):groups.setdefault((r['size'],r['category']),[]).append(i)
        groups=[np.array(groups[k]) for k in sorted(groups)];pairs={}
        for arm in ('autonomous','assisted'):
            head.load_state_dict(parent['head']);opt=torch.optim.Adam(head.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(optstate))
            rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['rng']);draws=hashlib.sha256()
            for step in range(1,1501):
                ix=np.array([int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(16)],np.int64)
                kx=rng.integers(len(old7),size=16,dtype=np.int64);jx=rng.integers(len(old9),size=16,dtype=np.int64);u=rng.random(16);nx=(u*len(fresh[arm])).astype(np.int64)
                draws.update(ix.tobytes()+kx.tobytes()+jx.tobytes()+u.tobytes())
                a,l,c,y=[torch.cat([v[ix],w[kx],z[jx],n[nx]]) for v,w,z,n in zip(*fixed,new[arm])]
                opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%250==0:
                    state=dict(head=copy.deepcopy(head.state_dict()),encoder=parent['encoder'],optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=step)
                    torch.save(state,OUT/f'latest-{arm}.pt');rec=dict(phase='training',arm=arm,step=step);write_json(OUT/'progress.json',rec);print(rec,flush=True)
            pairs[arm]=draws.hexdigest()
        assert pairs['autonomous']==pairs['assisted'];write_json(OUT/'pairing.json',pairs)
        candidates={'baseline':PARENT,'autonomous':OUT/'latest-autonomous.pt','assisted':OUT/'latest-assisted.pt'}
        chosen={a:dict(path=str(p),sha256=digest(p)) for a,p in candidates.items()};write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        for arm,path in candidates.items():
            head.load_state_dict(torch.load(path,weights_only=False)['head']);rows=[]
            for start in range(0,750,50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='autonomous_evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert frozen==tensor_hash(brain.state_dict()) and all(digest(p)==sha for p,sha in hashes.items())
        assert digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,brain_frozen=True,paired_fixed_samples=True,teacher_actions_at_test=0))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main()
