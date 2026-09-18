"""Larger supervised control and factorized full-connectome comparisons.

CNN control and ablations use separate budgets; only the four ablations are paired.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.backbone import Backbone, encode_visible
from experiments.capacity_probe import arrays, measure, write_json
from experiments.expressive_models import ConnectomePolicy, ConvPolicy, equivalent_loss
from experiments.train import visible, public_context
from minesweeper import Minesweeper
from solver import analyze


FLAGS = {'current':(False,False), 'encoder_only':(True,False),
         'internal_only':(False,True), 'both':(True,True)}


def make_model(name, seed, operator=None):
    torch.manual_seed(seed)
    if name == 'cnn':
        return ConvPolicy()
    encoder, internal = FLAGS[name]
    model = ConnectomePolicy(operator, channels=1, learned_encoder=encoder, internal_dynamics=internal)
    # All four arms start with exactly the same readout, despite different constructors.
    torch.manual_seed(seed+100)
    torch.nn.init.normal_(model.head.weight, std=.01/np.sqrt(model.head.in_features))
    torch.nn.init.zeros_(model.head.bias)
    return model


@torch.no_grad()
def evaluate_games(model, games, batch=16):
    model.eval()
    results, active, cursor = [], [], 0
    while cursor < len(games) or active:
        while cursor < len(games) and len(active) < batch:
            game = games[cursor]
            cursor += 1
            env = Minesweeper(game['seed'], game['size'], game['mines'])
            row = {**game, 'automatic':bool(env.won), 'clicks':0, 'safe_opportunities':0,
                   'safe_choices':0, 'guesses':0, 'known_mine_choices':0,
                   'death_with_safe_available':False, 'death_after_exact_half_min_risk':False}
            if env.done:
                results.append({**row, 'won':bool(env.won)})
            else:
                active.append((env,row))
        if not active:
            continue
        x, legal = encode_visible([visible(env) for env,_ in active])
        context = np.array([public_context(env.size,env.mine_count) for env,_ in active])
        z = model(torch.from_numpy(x),torch.from_numpy(context))
        actions = z.masked_fill(~torch.from_numpy(legal),-1e9).argmax(1).tolist()
        remaining = []
        for (env,row), action in zip(active,actions):
            info = analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
            native = (action//16)*env.size+action%16
            had_safe = bool(info.safe)
            row['safe_opportunities'] += int(had_safe)
            row['safe_choices'] += int(had_safe and native in info.safe)
            row['known_mine_choices'] += int(native in info.mines)
            row['guesses'] += int(not had_safe)
            is_exact_half = False
            if not had_safe and info.probability_method.startswith('exact'):
                probabilities = [info.mine_probability[int(i)] for i in np.flatnonzero(env.legal_mask())]
                is_exact_half = abs(min(probabilities)-.5)<1e-10 and abs(info.mine_probability[native]-.5)<1e-10
            env.step(native)
            row['clicks'] += 1
            if env.done:
                row['death_with_safe_available'] = bool(not env.won and had_safe)
                row['death_after_exact_half_min_risk'] = bool(not env.won and is_exact_half)
                results.append({**row,'won':bool(env.won)})
            else:
                remaining.append((env,row))
        active = remaining
    return sorted(results,key=lambda row:(row['size'],row['seed']))


def save_checkpoint(path, model, optimizer, rng, step, history, before, train_seconds):
    temp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),step=step,history=history,before=before,
                    train_seconds=train_seconds),temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('runs/scaled-learning-001/dataset.pkl'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=['cnn',*FLAGS],required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[20260924,20260925])
    parser.add_argument('--updates',type=int,required=True)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int,help='Total updates per invocation per model, for resume verification')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    sources = [Path(__file__),Path('experiments/scaled_data.py'),Path('experiments/expressive_models.py'),Path('experiments/capacity_probe.py'),
               Path('experiments/backbone.py'),Path('experiments/train.py'),Path('solver.py'),Path('minesweeper.py')]
    checkpoint = Path('runs/hybrid-001/checkpoint.pkl')
    manifest = dict(dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    protected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    seeds=args.seeds,variants=args.variants,updates=args.updates,batch=args.batch,
                    lr=.001,channels=1,torch=torch.__version__,
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    path = args.out/'manifest.json'
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != manifest:
            raise ValueError('Existing run requires --resume with identical config, data, protected checkpoint and source')
    else:
        if args.resume:
            raise FileNotFoundError('No existing manifest to resume')
        write_json(path,manifest)
        (args.out/'source').mkdir()
        for p in sources:
            (args.out/'source'/p.name).write_bytes(p.read_bytes())
    data = pickle.loads(args.data.read_bytes())
    # Store compact boards, not 10000 materialized floating-point one-hot states.
    groups = {}
    for index,row in enumerate(data['train']):
        groups.setdefault((row['size'],row['category']),[]).append(index)
    group_arrays = [np.array(groups[k]) for k in sorted(groups)]
    operator = None
    if any(v != 'cnn' for v in args.variants):
        operator = Backbone(sparse.load_npz('data/processed/graph.npz'),
                            'runs/curriculum-001/initial-expanded.npz',workers=4)
    results_path = args.out/'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for seed in args.seeds:
        for variant in args.variants:
            if any(r['seed']==seed and r['variant']==variant for r in results):
                continue
            model = make_model(variant,seed,operator)
            optimizer = torch.optim.Adam(model.parameters(),lr=.001)
            rng = np.random.default_rng(seed+1)
            history,step,train_seconds = [],0,0.
            state_path = args.out/f'{variant}-{seed}.pt'
            if args.resume and state_path.exists():
                saved = torch.load(state_path,weights_only=False)
                model.load_state_dict(saved['model'])
                optimizer.load_state_dict(saved['optimizer'])
                rng.bit_generator.state = saved['rng']
                torch.set_rng_state(saved['torch_rng'])
                history,step,train_seconds,before = saved['history'],saved['step'],saved['train_seconds'],saved['before']
            else:
                before = measure(model,data['holdout'])
            start_step = step
            started = time.monotonic()
            for step in range(step+1,args.updates+1):
                ids = [int(rng.choice(group_arrays[(step*args.batch+j)%len(group_arrays)])) for j in range(args.batch)]
                x,legal,context,labels = arrays([data['train'][i] for i in ids])
                optimizer.zero_grad(set_to_none=True)
                loss = equivalent_loss(model(x,context),legal,labels)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                if isinstance(model,ConnectomePolicy):
                    with torch.no_grad():
                        model.log_gain.clamp_(-1,1)
                stopping = args.stop_after is not None and step-start_step>=args.stop_after
                if step%100==0 or step==args.updates or stopping:
                    elapsed = train_seconds+time.monotonic()-started
                    progress = dict(phase='training',variant=variant,seed=seed,updates=step,target=args.updates,
                                    loss=float(loss.detach()),gradient_norm=float(norm),training_seconds=elapsed)
                    history.append(progress)
                    save_checkpoint(state_path,model,optimizer,rng,step,history,before,elapsed)
                    write_json(args.out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
                if stopping and step<args.updates:
                    if operator is not None:
                        operator.close()
                    return
            train_seconds += time.monotonic()-started
            write_json(args.out/'progress.json',dict(phase='evaluation',variant=variant,seed=seed,updates=step))
            result = dict(variant=variant,seed=seed,updates=step,batch=args.batch,history=history,
                          training_seconds=train_seconds,parameters=sum(p.numel() for p in model.parameters()),
                          before=before,holdout=measure(model,data['holdout']),
                          games=evaluate_games(model,data['games']))
            if isinstance(model,ConnectomePolicy):
                result['nonzero_gains'] = int(model.log_gain.detach().count_nonzero())
            results.append(result)
            write_json(results_path,results)
            print('FINISHED',variant,seed,result['holdout']['correct'],flush=True)
            del model,optimizer
    if operator is not None:
        operator.close()
    intact = hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['protected_sha256']
    if not intact:
        raise RuntimeError('Protected checkpoint changed')
    write_json(args.out/'completed.json',dict(models=len(results),protected_checkpoint_intact=intact,time=time.time()))


if __name__ == '__main__':
    main()
