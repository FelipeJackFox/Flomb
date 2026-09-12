"""Variable-board curriculum, preserving the pilot and resumable schedule state."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper
from curriculum import GROUPS, mixture, draw_board, encode, decode, reward_step

STRATA = [(5, 3), (7, 7), (9, 10), (12, 24), (16, 40), (16, 56)]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def evaluate(policy, count=32, seed_start=1_500_000_000):
    result = {}
    for i, (size, mines) in enumerate(STRATA):
        wins = initial_wins = clicks = 0
        fractions = []
        for j in range(count):
            seed = seed_start + i*100_000 + j
            rng = np.random.default_rng(seed + 10_000)
            env = Minesweeper(seed, size, mines)
            initial_wins += int(env.won)
            while not env.done:
                if policy is None:
                    action = int(rng.choice(np.flatnonzero(env.legal_mask())))
                else:
                    obs, mask = encode(env)
                    probs, _ = policy.forward(obs, mask)
                    action = decode(rng.choice(len(probs), p=probs), size)
                env.step(action)
                clicks += 1
            wins += int(env.won)
            fractions.append(float(np.sum(env.visible >= 0)/(size*size-mines)))
        result[f'{size}x{size}-{mines}'] = {
            'episodes': count, 'wins': wins, 'win_rate': wins/count,
            'automatic_wins': initial_wins,
            'decision_win_rate': (wins-initial_wins)/max(1, count-initial_wins),
            'mean_safe_fraction': float(np.mean(fractions)), 'clicks': clicks,
            'seed_start': seed_start+i*100_000,
        }
        print('EVAL', f'{size}x{size}-{mines}', json.dumps(result[f'{size}x{size}-{mines}']), flush=True)
    return result


def train_episode(policy, env, rng, baseline, activation_cache_mb=0):
    """Cache a bounded prefix; recompute the rest without changing update order.

    The cap covers additional activation/probability array payload, not total RSS.
    Zero preserves the original recompute-only execution.
    """
    if not np.isfinite(activation_cache_mb) or activation_cache_mb < 0:
        raise ValueError('activation_cache_mb must be finite and nonnegative')
    budget = int(activation_cache_mb * 1024**2)
    per_click = policy.activation_cache_bytes()
    used = cached_steps = 0
    trajectory = []
    while not env.done:
        obs, legal = encode(env)
        keep_cache = used + per_click <= budget
        probs, cache = policy.forward(obs, legal, cache=keep_cache)
        if keep_cache:
            used += per_click
            cached_steps += 1
        action = int(rng.choice(len(probs), p=probs))
        reward = reward_step(env, decode(action, env.size))
        trajectory.append((obs, legal, action, reward, (probs, cache) if keep_cache else None))
    gradients = [np.zeros_like(p) for p in policy.parameters()]
    ret = 0.
    for obs, legal, action, reward, stored in reversed(trajectory):
        ret += reward  # gamma=1: total shaping cannot outweigh a mine on larger boards.
        probs, cache = stored if stored is not None else policy.forward(obs, legal, cache=True)
        score = -probs.copy()
        score[action] += 1.
        logp = np.log(np.maximum(probs, 1e-30))
        entropy_gradient = probs * (float(probs @ logp)-logp)
        dlogits = (ret-baseline)*score + .02*entropy_gradient
        for g, new in zip(gradients, policy.backward(cache, dlogits)):
            g += new
    norm = policy.apply(gradients) if trajectory else 0.
    policy.last_episode_compute = {
        "cached_steps": cached_steps, "recomputed_steps": len(trajectory)-cached_steps,
        "activation_cache_bytes": used,
    }
    return ret, norm, len(trajectory)


def run(args):
    if args.sparse_workers < 1 or not np.isfinite(args.activation_cache_mb) or args.activation_cache_mb < 0:
        raise ValueError("Invalid compute configuration")
    out = Path(args.run)
    if args.total_episodes < 2 or args.chunk_episodes < 1 or args.eval_per_stratum < 1:
        raise ValueError('Invalid episode budget')
    if out.exists() and not args.resume:
        raise ValueError('Run exists; choose another run or --resume')
    out.mkdir(parents=True, exist_ok=True)
    # Refuse simultaneous writers, including after a restarted app.
    import fcntl
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(out/'live-process.json', {'pid': os.getpid(), 'started_at': time.time()})
    started = time.perf_counter()
    graph = sparse.load_npz('data/processed/graph.npz')
    policy = BrainPolicy(graph, sparse_workers=args.sparse_workers)
    rng = np.random.default_rng(args.seed)
    baseline = {g: 0. for g in GROUPS}
    counts, clicks, start, prior_seconds = Counter(), 0, 0, 0.
    if args.resume:
        cfg = json.loads((out/'config.json').read_text())
        state = json.loads((out/'state.json').read_text())
        if args.total_episodes != cfg['total_episodes'] or args.seed != cfg['seed']:
            raise ValueError('Resume must preserve curriculum budget and RNG seed')
        if args.eval_per_stratum != cfg['eval_per_stratum']:
            raise ValueError('Resume must preserve evaluation count')
        policy.load(out/state['checkpoint'])
        rng.bit_generator.state = state['rng']
        start, clicks = state['episode'], state['clicks']
        baseline, counts = state['baseline'], Counter(state['group_counts'])
        prior_seconds = state['training_wall_seconds']
    else:
        source = Path(args.initial)
        policy.load(source)
        policy.expand_canvas()
        # Save actual code for reproduction; original run/checkpoint remain untouched.
        snapshot = out/'source'
        snapshot.mkdir()
        sources = ['brain.py', 'minesweeper.py', 'curriculum.py', 'train_curriculum.py', 'requirements.txt']
        for name in sources:
            shutil.copy2(name, snapshot/name)
        cfg = vars(args) | {
            'initial_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_hashes': {x: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in sources},
            'canvas': 16, 'initial_probabilities': [.6, .3, .1], 'pure_large_at_fraction': .7,
            'groups': {'small': '5/7, density .08-.16', 'medium': '9/12, density .10-.19',
                       'large': '16, density .12-.22'},
            'reward': 'mine -1, win +1, +.25 * newly revealed / total safe; gamma=1',
            'baseline': 'separate moving baseline for each size group',
            'migration': 'old input/output embedded exactly; extra inputs only outside original 5x5; Adam moments reset',
            'dynamics': 'full retained MaleCNS sparse rate graph, three rounds, internal gains and readout trainable',
            'train_seed_range': [0, 1_000_000_000], 'validation_seed_start': 1_500_000_000,
            'final_seed_start': 2_000_000_000, 'evaluation_strata': STRATA,
            'graph_provenance_sha256': hashlib.sha256(Path('data/processed/provenance.json').read_bytes()).hexdigest(),
        }
        atomic_json(out/'config.json', cfg)
        policy.save(out/'initial-expanded.npz')
        before = {'random': evaluate(None, args.eval_per_stratum),
                  'initial': evaluate(policy, args.eval_per_stratum)}
        atomic_json(out/'evaluation-before.json', before)
    # Record per-invocation execution settings separately from immutable curriculum config.
    invocation = out / f'compute-{time.time_ns()}.json'
    atomic_json(invocation, {
        'sparse_workers': args.sparse_workers, 'activation_cache_mb': args.activation_cache_mb,
        'resumed_from_episode': start, 'pid': os.getpid(),
        'source_hashes': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ('brain.py', 'train_curriculum.py')},
    })
    reference_gain = np.load(out/'initial-expanded.npz')['log_gain']
    stop = min(args.total_episodes, start+args.chunk_episodes)
    if start > stop:
        raise ValueError('Checkpoint exceeds configured curriculum')
    # Each invocation has its own append log; interrupted tails never replace consolidated state.
    log_path = out/f'metrics-from-{start:06d}-{time.time_ns()}.jsonl'
    training_started = time.perf_counter()

    def checkpoint(episode):
        filename = f'checkpoint-{episode:06d}.npz'
        tmp = out/f'checkpoint-{episode:06d}.tmp.npz'
        policy.save(tmp)
        tmp.replace(out/filename)
        atomic_json(out/'state.json', {
            'episode': episode, 'target': args.total_episodes, 'checkpoint': filename,
            'rng': rng.bit_generator.state, 'baseline': baseline, 'group_counts': dict(counts),
            'clicks': clicks, 'training_wall_seconds': prior_seconds+time.perf_counter()-training_started,
        })

    checkpoint(start)
    for episode in range(start, stop):
        group, size, mines, probabilities = draw_board(rng, episode, args.total_episodes)
        seed = int(rng.integers(0, 1_000_000_000))
        env = Minesweeper(seed, size, mines)
        automatic_win = bool(env.won)
        ret, norm, steps = train_episode(policy, env, rng, baseline[group], args.activation_cache_mb)
        if steps:
            baseline[group] = .98*baseline[group]+.02*ret
        counts[group] += 1
        clicks += steps
        row = {'episode': episode+1, 'target': args.total_episodes, 'group': group,
               'size': size, 'mines': mines, 'seed': seed, 'mixture': probabilities.tolist(),
               'won': bool(env.won), 'automatic_win': automatic_win, 'return': ret,
               'safe_fraction': float(np.mean(env.visible >= 0)*size*size/(size*size-mines)),
               'steps': steps, 'total_clicks': clicks, 'gradient_norm': norm,
               'internal_gain_l2_change': float(np.linalg.norm(policy.log_gain-reference_gain)),
               'group_counts': dict(counts),
               'training_wall_seconds': prior_seconds+time.perf_counter()-training_started,
               'compute': policy.last_episode_compute,
               'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        with log_path.open('a') as f:
            f.write(json.dumps(row)+'\n')
        atomic_json(out/'progress.json', row)
        if (episode+1) % 16 == 0:
            print('TRAIN', json.dumps(row), flush=True)
        if (episode+1) % 256 == 0:
            checkpoint(episode+1)
    checkpoint(stop)
    if stop == args.total_episodes:
        final = {'trained': evaluate(policy, args.eval_per_stratum, 2_000_000_000),
                 'random': evaluate(None, args.eval_per_stratum, 2_000_000_000)}
        # Same fresh final boards for the migrated initial model, without selecting checkpoints on them.
        policy.load(out/'initial-expanded.npz')
        final['initial'] = evaluate(policy, args.eval_per_stratum, 2_000_000_000)
        atomic_json(out/'evaluation-final.json', final)
        atomic_json(out/'completed.json', {'episode': stop, 'completed_at': time.time(),
                                         'evaluation': 'evaluation-final.json'})
        print('COMPLETED', args.run, flush=True)
    else:
        print('CHUNK_COMPLETED', stop, 'of', args.total_episodes, flush=True)
    policy.close()
    print('INVOCATION_SECONDS', time.perf_counter()-started, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='runs/curriculum-001')
    p.add_argument('--initial', default='runs/pilot-001/checkpoint-002176.npz')
    p.add_argument('--total-episodes', type=int, default=50000)
    p.add_argument('--chunk-episodes', type=int, default=50000)
    p.add_argument('--eval-per-stratum', type=int, default=32)
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--sparse-workers', type=int, default=1)
    p.add_argument('--activation-cache-mb', type=float, default=0,
                   help='Maximum additional cached activation array MiB per episode; overflow recomputes')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
