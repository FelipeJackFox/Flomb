"""Independent DAgger, dueling Double QR-DQN+PER, and their hybrid.
Run with python -m experiments.train. Checkpoints are trusted-local pickle files.
"""
import argparse,copy,hashlib,json,os,pickle,time,resource
from collections import deque
from pathlib import Path
import numpy as np
from scipy import sparse
from minesweeper import Minesweeper
from curriculum import draw_board,decode,reward_step
from solver import analyze
from train_curriculum import STRATA,atomic_json
from experiments.backbone import Backbone,encode_visible
from experiments.qr_core import DuelingQuantileHead,PrioritizedReplay,quantile_huber

def visible(env):
    b=np.full((16,16),-2,np.int8);b[:env.size,:env.size]=env.visible.reshape(env.size,env.size)
    return b.reshape(-1)

def teacher(env):
    # Only public observation, legal actions, board size and total mines cross this boundary.
    info=analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
    candidates=sorted(info.safe)
    if not candidates:
        available=[i for i in np.flatnonzero(env.legal_mask()) if i not in info.mines]
        risk=min(info.mine_probability[i] for i in available)
        candidates=[i for i in available if info.mine_probability[i]<=risk+1e-12]
    targets=np.zeros(256,bool)
    for i in candidates:targets[(i//env.size)*16+i%env.size]=True
    confidence=1. if info.safe or info.probability_method.startswith('exact') else .2
    return targets,confidence

def supervised(z,mask,labels,confidence,weights):
    logits=z.mean(axis=2);masked=np.where(mask,logits,-1e9)
    p=np.exp(masked-masked.max(axis=1,keepdims=True));p*=mask;p/=p.sum(axis=1,keepdims=True)
    labels=labels.astype(np.float32);labels/=np.maximum(labels.sum(axis=1,keepdims=True),1)
    effective=confidence*weights
    loss=float(np.mean(-np.sum(labels*np.log(np.maximum(p,1e-12)),axis=1)*effective))
    dz=np.broadcast_to(((p-labels)*effective[:,None]/len(z)/z.shape[2])[:,:,None],z.shape).copy()
    return loss,dz

def public_context(size,mines):
    return np.array([size/16,mines/(size*size)],np.float32)

def greedy(backbone,head,b,size,mines):
    obs,mask=encode_visible([b]);f,_=backbone.forward(obs,False);z,_=head.forward(np.concatenate([f,public_context(size,mines)[None,:]],axis=1))
    return int(np.argmax(np.where(mask[0],z[0].mean(axis=1),-np.inf)))

def evaluate(backbone,head,count):
    result={}
    for group,(size,mines) in enumerate(STRATA):
        wins=automatic=clicks=0;fractions=[]
        for j in range(count):
            env=Minesweeper(2_000_000_000+group*100000+j,size,mines);automatic+=int(env.won)
            while not env.done:
                a=greedy(backbone,head,visible(env),size,mines);env.step(decode(a,size));clicks+=1
            wins+=int(env.won);fractions.append(float(np.mean(env.visible>=0)*size*size/(size*size-mines)))
        result[f'{size}x{size}-{mines}']={'episodes':count,'wins':wins,'automatic_wins':automatic,'mean_safe_fraction':float(np.mean(fractions)),'clicks':clicks,'teacher':False,'exploration':False,'seed_start':2_000_000_000+group*100000}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['dagger','qrdqn','hybrid'],required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--episodes',type=int,default=64)
    ap.add_argument('--curriculum-horizon',type=int,default=50000);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--capacity',type=int,default=8192);ap.add_argument('--warmup',type=int,default=32)
    ap.add_argument('--train-every',type=int,default=4);ap.add_argument('--target-every',type=int,default=100)
    ap.add_argument('--n-step',type=int,default=3);ap.add_argument('--gamma',type=float,default=.99)
    ap.add_argument('--quantiles',type=int,default=16);ap.add_argument('--eval-per-stratum',type=int,default=4)
    ap.add_argument('--stop-after',type=int,default=None,help='Maximum additional episodes this invocation; resume preserves schedule')
    ap.add_argument('--seed',type=int,default=20260915);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--init',type=Path,default=Path('runs/curriculum-001/initial-expanded.npz'))
    args=ap.parse_args()
    for key in ('workers','episodes','curriculum_horizon','batch','capacity','warmup','train_every','target_every','n_step','quantiles','eval_per_stratum'):
        if getattr(args,key)<1:ap.error(key+' must be positive')
    if args.episodes>args.curriculum_horizon or args.batch>args.capacity:ap.error('invalid budget/buffer')
    if args.stop_after is not None and args.stop_after<1:ap.error('stop-after must be positive')
    run=args.run;run.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','run','stop_after','workers')}
    config['initial_sha256']=hashlib.sha256(args.init.read_bytes()).hexdigest()
    config['public_context']='board size and total mine density; no hidden locations'
    config['algorithm']='DAgger / dueling Double QR-DQN with PER and n-step / hybrid; not full Rainbow (no NoisyNet)'
    if (run/'config.json').exists():
        if not args.resume:raise RuntimeError('Run exists; use --resume')
        old=json.loads((run/'config.json').read_text())
        if old!=config:raise ValueError('Resume configuration differs; use a new run')
    else:
        if args.resume:raise FileNotFoundError('No checkpoint to resume')
        atomic_json(run/'config.json',config)
        source=run/'source';source.mkdir(exist_ok=True)
        sources=['experiments/train.py','experiments/backbone.py','experiments/qr_core.py','curriculum.py','minesweeper.py','solver.py']
        for name in sources:(source/name.replace('/','_')).write_bytes(Path(name).read_bytes())
        atomic_json(run/'source-sha256.json',{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    atomic_json(run/f'compute-{time.time_ns()}.json',{'workers':args.workers,'batch':args.batch,'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['experiments/train.py','experiments/backbone.py','experiments/qr_core.py']}})
    graph=sparse.load_npz('data/processed/graph.npz');brain=Backbone(graph,args.init,workers=args.workers);target_brain=brain.target()
    head=DuelingQuantileHead(features=len(brain.output_index)+2,actions=256,quantiles=args.quantiles,seed=args.seed)
    target_head=DuelingQuantileHead(features=len(brain.output_index)+2,actions=256,quantiles=args.quantiles,seed=args.seed)
    target_head.load_state_dict(head.state_dict())
    replay=PrioritizedReplay(args.capacity,seed=args.seed+3,alpha=0. if args.mode=='dagger' else .6)
    board_rng=np.random.default_rng(args.seed);action_rng=np.random.default_rng(args.seed+1)
    ep=steps=updates=0;train_seconds=0.;last_loss={};pending=deque()
    if args.resume:
        with (run/'checkpoint.pkl').open('rb') as f:s=pickle.load(f)
        for obj,key in [(brain,'brain'),(target_brain,'target_brain'),(head,'head'),(target_head,'target_head'),(replay,'replay')]:obj.load_state_dict(s[key])
        ep,steps,updates,train_seconds=s['episode'],s['steps'],s['updates'],s['train_seconds']
        last_loss=s.get('last_loss',{});board_rng.bit_generator.state=s['board_rng'];action_rng.bit_generator.state=s['action_rng']
    start=time.monotonic();atomic_json(run/'live-process.json',{'pid':os.getpid(),'mode':args.mode,'started_at':time.time()})
    def checkpoint():
        data={'brain':brain.state_dict(),'target_brain':target_brain.state_dict(),'head':head.state_dict(),'target_head':target_head.state_dict(),'replay':replay.state_dict(),'episode':ep,'steps':steps,'updates':updates,'train_seconds':train_seconds+time.monotonic()-start,'last_loss':last_loss,'board_rng':board_rng.bit_generator.state,'action_rng':action_rng.bit_generator.state}
        temp=run/'checkpoint.tmp';
        with temp.open('wb') as f:pickle.dump(data,f,protocol=5)
        temp.replace(run/'checkpoint.pkl');atomic_json(run/'state.json',{'episode':ep,'steps':steps,'updates':updates,'checkpoint':'checkpoint.pkl'})
    def enqueue_transition(item):
        pending.append(item)
        if len(pending)>=args.n_step or item['done']:
            while pending and (len(pending)>=args.n_step or item['done']):
                first=pending[0].copy();trajectory=list(pending)[:args.n_step]
                first['reward']=sum(args.gamma**i*t['reward'] for i,t in enumerate(trajectory))
                first['next']=trajectory[-1]['next'];first['done']=trajectory[-1]['done'];first['discount']=args.gamma**len(trajectory)
                replay.add(first);pending.popleft()
    def update():
        nonlocal updates,last_loss
        beta=min(1.,.4+.6*ep/args.curriculum_horizon)
        items,indices,weights=replay.sample(args.batch,beta=beta)
        x,mask=encode_visible([t['visible'] for t in items]);features,cache=brain.forward(x);context=np.array([t['context'] for t in items]);z,hcache=head.forward(np.concatenate([features,context],axis=1))
        dz=np.zeros_like(z);loss_q=loss_teacher=0.;priorities=np.ones(len(items),np.float32)
        if args.mode!='dagger':
            nx,nmask=encode_visible([t['next'] for t in items]);nonterminal=np.array([not t['done'] for t in items]);dest=np.zeros((len(items),args.quantiles),np.float32)
            if nonterminal.any():
                nf,_=brain.forward(nx[nonterminal],False);online,_=head.forward(np.concatenate([nf,context[nonterminal]],axis=1))
                actions=np.argmax(np.where(nmask[nonterminal],online.mean(axis=2),-np.inf),axis=1)
                tf,_=target_brain.forward(nx[nonterminal],False);tz,_=target_head.forward(np.concatenate([tf,context[nonterminal]],axis=1))
                dest[nonterminal]=tz[np.arange(len(actions)),actions]
            target=np.array([t['reward'] for t in items],np.float32)[:,None]+np.array([t['discount'] for t in items],np.float32)[:,None]*dest
            chosen=np.array([t['action'] for t in items]);loss_q,dchosen,priorities=quantile_huber(z[np.arange(len(items)),chosen],target,weights)
            dz[np.arange(len(items)),chosen]=dchosen
        if args.mode!='qrdqn':
            labels=np.array([np.unpackbits(t['teacher'],count=256) for t in items],np.float32)
            confidence=np.array([t['confidence'] for t in items],np.float32)
            loss_teacher,dt=supervised(z,mask,labels,confidence,weights)
            strength=1. if args.mode=='dagger' else max(.1,1-ep/(.6*args.episodes))
            dz+=strength*dt
            if args.mode=='dagger':priorities=np.ones(len(items),np.float32) # DAgger uniform data, no TD prioritization.
        df,grads=head.backward(hcache,dz);dg=brain.backward(cache,df[:,:len(brain.output_index)]);head.apply(grads);brain.apply(dg)
        replay.update_priorities(indices,np.asarray(priorities)+1e-5);updates+=1
        if updates%args.target_every==0:
            target_brain.log_gain=brain.log_gain.copy();target_head.load_state_dict(head.state_dict())
        last_loss={'qr':float(loss_q),'teacher':float(loss_teacher)}
    if not args.resume:checkpoint()
    stop=min(args.episodes,ep+args.stop_after) if args.stop_after else args.episodes
    while ep<stop:
        group,size,mines,mix=draw_board(board_rng,ep,args.curriculum_horizon)
        env=Minesweeper(int(board_rng.integers(0,1_000_000_000)),size,mines)
        automatic=env.won;episode_steps=0;teacher_actions=0
        beta=max(0.,1-ep/(.6*args.episodes)) if args.mode!='qrdqn' else 0.
        epsilon=max(.05,1-ep/(.5*args.episodes))
        while not env.done:
            before=visible(env);labels=np.zeros(256,bool);confidence=0.
            if args.mode!='qrdqn':labels,confidence=teacher(env)
            if args.mode!='qrdqn' and action_rng.random()<beta:
                action=int(action_rng.choice(np.flatnonzero(labels)));teacher_actions+=1
            elif args.mode!='dagger' and action_rng.random()<epsilon:
                action=int(action_rng.choice(np.flatnonzero(before==-1)))
            else:action=greedy(brain,head,before,size,mines)
            reward=reward_step(env,decode(action,size));steps+=1;episode_steps+=1
            enqueue_transition({'visible':before,'next':visible(env),'action':action,'reward':reward,'done':env.done,'teacher':np.packbits(labels),'confidence':confidence,'context':public_context(size,mines)})
            if steps>=args.warmup and len(replay)>=args.batch and steps%args.train_every==0:update()
        assert not pending
        ep+=1
        progress={'mode':args.mode,'workers':args.workers,'episode':ep,'target':args.episodes,'curriculum_horizon':args.curriculum_horizon,'steps':steps,'updates':updates,'size':size,'mines':mines,'won':env.won,'automatic_win':automatic,'episode_steps':episode_steps,'teacher_actions':teacher_actions,'teacher_beta':beta,'epsilon':epsilon if args.mode!='dagger' else 0.,'loss':last_loss,'training_seconds':train_seconds+time.monotonic()-start,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        atomic_json(run/'progress.json',progress)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        if ep%16==0 or ep==stop:checkpoint();print(json.dumps(progress),flush=True)
    if ep<args.episodes:
        atomic_json(run/'paused.json',{'episode':ep,'reason':'invocation limit reached; resume authorized explicitly by caller'})
        return
    (run/'paused.json').unlink(missing_ok=True)
    atomic_json(run/'evaluation-final.json',evaluate(brain,head,args.eval_per_stratum))
    atomic_json(run/'completed.json',{'episodes':ep,'updates':updates,'checkpoint':'checkpoint.pkl','evaluation':'evaluation-final.json','kind':'bounded pilot; not evidence of superiority','time':time.time()})
if __name__=='__main__':main()
