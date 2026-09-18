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
from experiments.behavior_schedule import load_schedule,probabilities
from experiments.game_metrics import analyze_move,summarize_moves
from experiments.curriculum_control import sample_board,applied_control
from experiments.qr_core import DuelingQuantileHead,PrioritizedReplay,quantile_huber

def visible(env):
    b=np.full((16,16),-2,np.int8);b[:env.size,:env.size]=env.visible.reshape(env.size,env.size)
    return b.reshape(-1)

def teacher(env, info=None):
    # Only public observation, legal actions, board size and total mines cross this boundary.
    if info is None:info=analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
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

def metric_summary(moves,won,automatic,complete=True):
    return summarize_moves(moves if complete else [*moves,{'metrics_sampled':True}],won,automatic)

def aggregate_metrics(summaries):
    ratios={'safe_take_rate','exact_coverage','mean_excess_risk','revealed_per_click','guess_survival_residual'}
    if not summaries:return {}
    out={k:sum(row[k] for row in summaries) for k in summaries[0] if k not in ratios}
    for name,numerator,denominator in [('safe_take_rate','safe_taken','safe_opportunities'),('exact_coverage','exact_moves','analyzed_moves'),('mean_excess_risk','excess_risk_sum','exact_moves'),('revealed_per_click','revealed_cells','reveal_observations')]:
        out[name]=out[numerator]/out[denominator] if out[denominator] else None
    n=out['exact_guess_observations'];out['guess_survival_residual']=(out['exact_guess_survivals']-out['exact_guess_expected_survivals'])/n if n else None
    return out

def evaluate(backbone,head,count):
    result={}
    for group,(size,mines) in enumerate(STRATA):
        wins=automatic=clicks=0;fractions=[];summaries=[]
        for j in range(count):
            env=Minesweeper(2_000_000_000+group*100000+j,size,mines);initial=bool(env.won);automatic+=int(initial);moves=[]
            while not env.done:
                a=decode(greedy(backbone,head,visible(env),size,mines),size)
                record=analyze_move(env.visible,size,mines,a);before=int(np.sum(env.visible>=0))
                env.step(a);clicks+=1
                record.update(survived=not env.done or bool(env.won),revealed_delta=int(np.sum(env.visible>=0))-before);moves.append(record)
            wins+=int(env.won);fractions.append(float(np.sum(env.visible>=0)/(size*size-mines)))
            summaries.append(metric_summary(moves,env.won,initial))
        result[f'{size}x{size}-{mines}']={'episodes':count,'wins':wins,'automatic_wins':automatic,'mean_safe_fraction':float(np.mean(fractions)),'clicks':clicks,'teacher':False,'exploration':False,'seed_start':2_000_000_000+group*100000,'game_metrics':aggregate_metrics(summaries)}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['dagger','qrdqn','hybrid'],required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--episodes',type=int,default=64)
    ap.add_argument('--curriculum-horizon',type=int,default=50000);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--capacity',type=int,default=8192);ap.add_argument('--warmup',type=int,default=32)
    ap.add_argument('--train-every',type=int,default=4);ap.add_argument('--target-every',type=int,default=100)
    ap.add_argument('--n-step',type=int,default=3);ap.add_argument('--gamma',type=float,default=.99)
    ap.add_argument('--quantiles',type=int,default=16);ap.add_argument('--eval-per-stratum',type=int,default=4)
    ap.add_argument('--metrics-every',type=int,default=1,help='Analyze every Nth whole training episode; no action subsampling')
    ap.add_argument('--eval-every',type=int,default=256,help='Teacher-free evaluation interval in completed episodes')
    ap.add_argument('--stop-after',type=int,default=None,help='Maximum additional episodes this invocation; resume preserves schedule')
    ap.add_argument('--seed',type=int,default=20260915);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--init',type=Path,default=Path('runs/curriculum-001/initial-expanded.npz'))
    args=ap.parse_args()
    for key in ('metrics_every','eval_every','workers','episodes','curriculum_horizon','batch','capacity','warmup','train_every','target_every','n_step','quantiles','eval_per_stratum'):
        if getattr(args,key)<1:ap.error(key+' must be positive')
    if args.episodes>args.curriculum_horizon or args.batch>args.capacity:ap.error('invalid budget/buffer')
    if args.stop_after is not None and args.stop_after<1:ap.error('stop-after must be positive')
    run=args.run;run.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','run','stop_after','workers','metrics_every','eval_every')}
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
        sources=['experiments/train.py','experiments/backbone.py','experiments/qr_core.py','experiments/game_metrics.py','experiments/curriculum_control.py','experiments/behavior_schedule.py','curriculum.py','minesweeper.py','solver.py']
        for name in sources:(source/name.replace('/','_')).write_bytes(Path(name).read_bytes())
        atomic_json(run/'source-sha256.json',{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    invocation=str(time.time_ns());source=run/'invocations'/invocation;source.mkdir(parents=True)
    files=['experiments/train.py','experiments/backbone.py','experiments/qr_core.py','experiments/game_metrics.py','experiments/curriculum_control.py','experiments/behavior_schedule.py','curriculum.py','minesweeper.py','solver.py','train_curriculum.py']
    for name in files:(source/name.replace('/','_')).write_bytes(Path(name).read_bytes())
    atomic_json(run/f'compute-{invocation}.json',{'workers':args.workers,'batch':args.batch,'metrics_every':args.metrics_every,'eval_every':args.eval_every,'source_directory':str(source),'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}})
    atomic_json(run/'capabilities.json',{'runtime_curriculum':True,'game_metrics':True,'metrics_every':args.metrics_every,'eval_every':args.eval_every,'invocation_id':invocation})
    graph=sparse.load_npz('data/processed/graph.npz');brain=Backbone(graph,args.init,workers=args.workers);target_brain=brain.target()
    head=DuelingQuantileHead(features=len(brain.output_index)+2,actions=256,quantiles=args.quantiles,seed=args.seed)
    target_head=DuelingQuantileHead(features=len(brain.output_index)+2,actions=256,quantiles=args.quantiles,seed=args.seed)
    target_head.load_state_dict(head.state_dict())
    replay=PrioritizedReplay(args.capacity,seed=args.seed+3,alpha=0. if args.mode=='dagger' else .6)
    board_rng=np.random.default_rng(args.seed);action_rng=np.random.default_rng(args.seed+1)
    ep=steps=updates=0;train_seconds=0.;last_loss={};last_gradient=None;pending=deque()
    if args.resume:
        with (run/'checkpoint.pkl').open('rb') as f:s=pickle.load(f)
        for obj,key in [(brain,'brain'),(target_brain,'target_brain'),(head,'head'),(target_head,'target_head'),(replay,'replay')]:obj.load_state_dict(s[key])
        ep,steps,updates,train_seconds=s['episode'],s['steps'],s['updates'],s['train_seconds']
        last_loss=s.get('last_loss',{});last_gradient=s.get('last_gradient');board_rng.bit_generator.state=s['board_rng'];action_rng.bit_generator.state=s['action_rng']
    behavior_schedule=load_schedule(run)
    if behavior_schedule is not None:
        atomic_json(source/'behavior-schedule.json',behavior_schedule)
        atomic_json(run/'behavior-applied.json',{'invocation_id':invocation,'start_episode':ep+1,'schedule':behavior_schedule})
    start=time.monotonic();atomic_json(run/'live-process.json',{'pid':os.getpid(),'mode':args.mode,'started_at':time.time(),'invocation_id':invocation})
    def checkpoint():
        data={'brain':brain.state_dict(),'target_brain':target_brain.state_dict(),'head':head.state_dict(),'target_head':target_head.state_dict(),'replay':replay.state_dict(),'episode':ep,'steps':steps,'updates':updates,'train_seconds':train_seconds+time.monotonic()-start,'last_loss':last_loss,'last_gradient':last_gradient,'board_rng':board_rng.bit_generator.state,'action_rng':action_rng.bit_generator.state}
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
        nonlocal updates,last_loss,last_gradient
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
        df,grads=head.backward(hcache,dz);dg=brain.backward(cache,df[:,:len(brain.output_index)]);head_norm=head.apply(grads);brain_norm=brain.apply(dg);last_gradient={'head':float(head_norm),'backbone':float(brain_norm),'combined':float(np.hypot(head_norm,brain_norm))}
        replay.update_priorities(indices,np.asarray(priorities)+1e-5);updates+=1
        if updates%args.target_every==0:
            target_brain.log_gain=brain.log_gain.copy();target_head.load_state_dict(head.state_dict())
        last_loss={'qr':float(loss_q),'teacher':float(loss_teacher)}
    if not args.resume:checkpoint()
    stop=min(args.episodes,ep+args.stop_after) if args.stop_after else args.episodes
    latest_evaluation=None
    while ep<stop:
        group,size,mines,mix,control=sample_board(board_rng,ep,args.curriculum_horizon,run)
        applied_control(run,control,ep+1)
        env=Minesweeper(int(board_rng.integers(0,1_000_000_000)),size,mines)
        automatic=env.won;episode_steps=0;teacher_actions=0;episode_return=0.;metrics_seconds=0.;moves=[];controllers={k:[] for k in ('teacher','random','policy')};controller_counts={k:0 for k in controllers};measured=ep%args.metrics_every==0
        beta,epsilon=probabilities(behavior_schedule,ep,args.episodes,args.mode)
        while not env.done:
            before=visible(env);labels=np.zeros(256,bool);confidence=0.
            info=None
            if args.mode!='qrdqn':
                info=analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
                labels,confidence=teacher(env,info)
            if args.mode!='qrdqn' and action_rng.random()<beta:
                action=int(action_rng.choice(np.flatnonzero(labels)));teacher_actions+=1;controller='teacher'
            elif args.mode!='dagger' and action_rng.random()<epsilon:
                action=int(action_rng.choice(np.flatnonzero(before==-1)));controller='random'
            else:action=greedy(brain,head,before,size,mines);controller='policy'
            native=decode(action,size);record=None;revealed_before=int(np.sum(env.visible>=0))
            if measured:
                metric_start=time.monotonic();record=analyze_move(env.visible,size,mines,native,analysis=info);metrics_seconds+=time.monotonic()-metric_start
            reward=reward_step(env,native);steps+=1;episode_steps+=1;episode_return+=reward;controller_counts[controller]+=1
            if record is not None:
                record.update(survived=not env.done or bool(env.won),revealed_delta=int(np.sum(env.visible>=0))-revealed_before)
                moves.append(record);controllers[controller].append(record)
            enqueue_transition({'visible':before,'next':visible(env),'action':action,'reward':reward,'done':env.done,'teacher':np.packbits(labels),'confidence':confidence,'context':public_context(size,mines)})
            if steps>=args.warmup and len(replay)>=args.batch and steps%args.train_every==0:update()
        assert not pending
        ep+=1
        metric_start=time.monotonic()
        game_metrics=metric_summary(moves,env.won,automatic,measured)
        controller_metrics={k:metric_summary(v,env.won,automatic,measured and len(v)==episode_steps) for k,v in controllers.items() if controller_counts[k]}
        metrics_seconds+=time.monotonic()-metric_start
        progress={'behavior_schedule_version':behavior_schedule['version'] if behavior_schedule else 0,'behavior_schedule_start':behavior_schedule.get('start_episode') if behavior_schedule else None,'invocation_id':invocation,'return':float(episode_return),'safe_fraction':float(np.sum(env.visible>=0)/(size*size-mines)),'game_metrics':game_metrics,'metrics_measured':measured,'controller_metrics':controller_metrics,'controller_actions':controller_counts,'gradient_norm':last_gradient['combined'] if last_gradient else None,'gradient_norms':last_gradient,'timing':{'metrics_seconds':metrics_seconds},'timing_metrics_seconds':metrics_seconds,'curriculum_mode':control['mode'],'curriculum_revision':control['revision'],'curriculum_control':control,'mixture':mix.tolist(),'group':group,'mode':args.mode,'workers':args.workers,'episode':ep,'target':args.episodes,'curriculum_horizon':args.curriculum_horizon,'steps':steps,'updates':updates,'size':size,'mines':mines,'won':env.won,'automatic_win':automatic,'episode_steps':episode_steps,'teacher_actions':teacher_actions,'teacher_beta':beta,'epsilon':epsilon if args.mode!='dagger' else 0.,'loss':last_loss,'training_seconds':train_seconds+time.monotonic()-start,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        atomic_json(run/'progress.json',progress)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        if ep%16==0 or ep==stop or ep%args.eval_every==0:checkpoint();print(json.dumps(progress),flush=True)
        if ep%args.eval_every==0:
            evaluation={'episode':ep,'invocation_id':invocation,'results':evaluate(brain,head,args.eval_per_stratum)}
            latest_evaluation=evaluation
            with (run/'evaluations.jsonl').open('a') as f:f.write(json.dumps(evaluation)+'\n')
    if ep<args.episodes:
        atomic_json(run/'paused.json',{'episode':ep,'reason':'invocation limit reached; resume authorized explicitly by caller'})
        return
    (run/'paused.json').unlink(missing_ok=True)
    atomic_json(run/'evaluation-final.json',latest_evaluation['results'] if latest_evaluation and latest_evaluation['episode']==ep else evaluate(brain,head,args.eval_per_stratum))
    atomic_json(run/'completed.json',{'episodes':ep,'updates':updates,'checkpoint':'checkpoint.pkl','evaluation':'evaluation-final.json','kind':'training run; held-out evaluation required for quality claims','time':time.time()})
if __name__=='__main__':main()
