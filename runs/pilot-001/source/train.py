"""Bounded local REINFORCE pilot, with internal gains and Adam checkpointing."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper


def evaluate(policy, seeds, action_seed=123456):
    rng = np.random.default_rng(action_seed)
    wins, clicks, cleared, initial_wins = 0, 0, 0, 0
    for seed in seeds:
        env = Minesweeper(int(seed))
        initial_wins += int(env.won)
        while not env.done:
            legal = env.legal_mask()
            if policy is None:
                action = rng.choice(np.flatnonzero(legal))
            else:
                probs, _ = policy.forward(env.observation(), legal)
                action = rng.choice(len(probs), p=probs)
            env.step(action)
            clicks += 1
        wins += int(env.won)
        cleared += int(np.sum(env.visible >= 0))
    n = len(seeds)
    return dict(episodes=n, wins=wins, win_rate=wins/n, agent_clicks=clicks,
                mean_safe_revealed=cleared/n, initial_auto_wins=initial_wins,
                decision_win_rate=(wins-initial_wins)/max(1, n-initial_wins))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=128)
    p.add_argument('--eval-episodes', type=int, default=32)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--run', default='runs/pilot-001')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--freeze-internal', action='store_true')
    args = p.parse_args()
    out = Path(args.run)
    if args.episodes < 1 or args.eval_episodes < 1:
        raise ValueError('Episode counts must be positive')
    if out.exists() and not args.resume:
        raise ValueError('Run already exists; use --resume or a new --run')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    graph = sparse.load_npz('data/processed/graph.npz')
    policy = BrainPolicy(graph)
    rng = np.random.default_rng(args.seed)
    eval_seeds = np.arange(1_000_000_000, 1_000_000_000 + args.eval_episodes)
    running_baseline, start_episode, total_clicks = 0., 0, 0
    if args.resume:
        state = json.loads((out / 'state.json').read_text())
        config = json.loads((out / 'config.json').read_text())
        if config['freeze_internal'] != args.freeze_internal or config['seed'] != args.seed:
            raise ValueError('Resume must preserve seed and internal-freeze setting')
        policy.load(out / state['checkpoint'])
        rng.bit_generator.state = state['rng']
        running_baseline, start_episode = state['baseline'], state['episode']
        total_clicks = state['total_clicks']
    else:
        config = vars(args) | {'algorithm': 'episodic REINFORCE, reward-to-go, moving baseline, entropy bonus, Adam',
                             'board': {'size': 5, 'mines': 3, 'first_click': 'automatic safe center'},
                             'input': 'fixed random distributed projection of visible one-hot board; artificial interface',
                             'output': '1024 fixed random receiving neurons -> 25 masked action logits',
                             'internal_learning': '166700 per-presynaptic gains; outgoing edge weights tied per neuron',
                             'dynamics': 'rate network, tanh, 3 recurrent propagation rounds, reset at each decision',
                             'eval_seed_start': int(eval_seeds[0]),
                             'source_hashes': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                               for name in ('train.py', 'brain.py', 'minesweeper.py', 'prepare_data.py')},
                             'provenance_sha256': hashlib.sha256(Path('data/processed/provenance.json').read_bytes()).hexdigest()}
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        policy.save(out / 'initial.npz')
        before = {'random': evaluate(None, eval_seeds), 'initial': evaluate(policy, eval_seeds)}
        (out / 'evaluation-before.json').write_text(json.dumps(before, indent=2))
        print('BEFORE', json.dumps(before), flush=True)
    initial_gain = policy.log_gain.copy()

    def checkpoint(episode):
        filename = f'checkpoint-{episode:06d}.npz'
        policy.save(out / filename)
        state = {'episode': episode, 'baseline': running_baseline,
                 'rng': rng.bit_generator.state, 'total_clicks': total_clicks,
                 'checkpoint': filename}
        tmp = out / 'state.json.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / 'state.json')

    for episode in range(start_episode, start_episode + args.episodes):
        # Training boards occupy a seed range disjoint from evaluation boards.
        board_seed = int(rng.integers(0, 1_000_000_000))
        env = Minesweeper(board_seed)
        trajectory = []
        while not env.done:
            probs, cache = policy.forward(env.observation(), env.legal_mask(), cache=True)
            action = int(rng.choice(len(probs), p=probs))
            reward = env.step(action)
            trajectory.append((probs, cache, action, reward))
            total_clicks += 1
        gradients = [np.zeros_like(param) for param in policy.parameters()]
        ret = 0.
        for probs, cache, action, reward in reversed(trajectory):
            ret = reward + .97 * ret
            score = -probs.copy()
            score[action] += 1
            # Gradient ascent on expected return + 0.01 * policy entropy.
            logp = np.log(np.maximum(probs, 1e-30))
            entropy_gradient = probs * (float(probs @ logp) - logp)
            dlogits = (ret - running_baseline) * score + .01 * entropy_gradient
            for g, new in zip(gradients, policy.backward(cache, dlogits)):
                g += new
        norm = 0.
        if trajectory:
            # Episode-level policy-gradient sum; no length-dependent objective reweighting.
            norm = policy.apply(gradients, freeze_internal=args.freeze_internal)
            running_baseline = .95 * running_baseline + .05 * ret
        del trajectory
        row = {'episode': episode + 1, 'board_seed': board_seed, 'won': bool(env.won),
               'return': ret, 'safe_revealed': int(np.sum(env.visible >= 0)),
               'total_clicks': total_clicks, 'gradient_norm': norm,
               'internal_gain_l2_change_this_run': float(np.linalg.norm(policy.log_gain-initial_gain)),
               'elapsed_seconds': time.perf_counter()-started,
               'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if (episode + 1) % 8 == 0:
            print('TRAIN', json.dumps(row), flush=True)
        if (episode + 1) % 32 == 0:
            checkpoint(episode + 1)
    checkpoint(start_episode + args.episodes)
    after = evaluate(policy, eval_seeds)
    (out / f'evaluation-after-{start_episode + args.episodes:06d}.json').write_text(json.dumps(after, indent=2))
    print('AFTER', json.dumps(after), flush=True)
    print('COMPLETED', args.run, 'seconds', time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    main()
