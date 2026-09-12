"""Independent paired benchmark; never loads a connectome or training checkpoint."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from minesweeper import Minesweeper
from solver import choose

STRATA = [(5, 3), (7, 7), (9, 10), (12, 24), (16, 40), (16, 56)]


def wilson(wins, n):
    if not n:
        return None
    z = 1.96
    p = wins/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius), float(center+radius)]


def episode(policy, seed, size, mines, exact_limit=12, replay=False):
    env = Minesweeper(seed, size, mines)
    rng = np.random.default_rng(seed + 100_000_000)
    automatic = bool(env.won)
    clicks, safe_clicks, guesses, successful_guesses = 0, 0, 0, 0
    methods = Counter()
    events = []
    while not env.done:
        # Solver receives public values only. No env reference crosses this boundary.
        if policy == 'solver':
            action, info = choose(env.observation(), env.legal_mask(), size, mines, rng, exact_limit)
            kind = info.classify(action)
            method = info.probability_method
            risk = info.mine_probability[action]
        else:
            action = int(rng.choice(np.flatnonzero(env.legal_mask())))
            kind, method, risk = 'random', 'uniform_legal_action', None
        before = env.visible.tolist() if replay else None
        reward = env.step(action)
        clicks += 1
        safe_clicks += int(kind == 'proven_safe')
        guesses += int(kind == 'guess')
        successful_guesses += int(kind == 'guess' and reward != -1)
        methods[method] += 1
        if replay:
            events.append({'actor': policy, 'visible_before': before, 'action': action,
                           'decision': kind, 'mine_probability': risk, 'probability_method': method,
                           'visible_after': env.visible.tolist(), 'hit_mine': reward == -1,
                           'won': bool(env.won)})
    result = {'seed': seed, 'won': bool(env.won), 'automatic_win': automatic,
              'clicks': clicks, 'proven_safe_clicks': safe_clicks, 'guesses': guesses,
              'successful_guesses': successful_guesses, 'methods': dict(methods),
              'safe_fraction': float(np.sum(env.visible >= 0)/(size*size-mines))}
    if replay:
        result.update(actor=policy, size=size, mine_count=mines, events=events,
                      neural_activity=None, note='Reference solver, not the fly neural policy.')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=192)
    p.add_argument('--seed-start', type=int, default=3_000_000_000)
    p.add_argument('--exact-limit', type=int, default=12)
    p.add_argument('--output', default='reference/results/benchmark.json')
    p.add_argument('--replay-output', default='reference/results/solver-replay.json')
    args = p.parse_args()
    if args.episodes < 1 or not 0 <= args.exact_limit <= 18:
        p.error('episodes >= 1 and exact-limit in 0..18 required')
    start = time.monotonic()
    output = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': vars(args),
              'strata': {}, 'caveat': 'Exact probabilities only below frontier cap; fallback is a density heuristic. No neural policy evaluated.'}
    for si, (size, mines) in enumerate(STRATA):
        rows = {}
        paired = {}
        for policy in ('random', 'solver'):
            runs = [episode(policy, args.seed_start + si*100_000 + j, size, mines, args.exact_limit)
                    for j in range(args.episodes)]
            wins = sum(r['won'] for r in runs)
            auto = sum(r['automatic_win'] for r in runs)
            n = args.episodes-auto
            rows[policy] = {'episodes': args.episodes, 'wins': wins, 'automatic_wins': auto,
                            'decision_wins': wins-auto, 'decision_episodes': n,
                            'decision_win_rate': (wins-auto)/n if n else None,
                            'decision_win_rate_wilson95': wilson(wins-auto, n),
                            'mean_safe_fraction': float(np.mean([r['safe_fraction'] for r in runs])),
                            'clicks': sum(r['clicks'] for r in runs),
                            'proven_safe_clicks': sum(r['proven_safe_clicks'] for r in runs),
                            'guesses': sum(r['guesses'] for r in runs),
                            'successful_guesses': sum(r['successful_guesses'] for r in runs),
                            'episodes_detail': runs}
            paired[policy] = runs
        rows['paired'] = {'solver_only_wins': sum(s['won'] and not r['won'] for s, r in zip(paired['solver'], paired['random'])),
                          'random_only_wins': sum(r['won'] and not s['won'] for s, r in zip(paired['solver'], paired['random']))}
        key = f'{size}x{size}-{mines}'
        output['strata'][key] = rows
        output['elapsed_seconds'] = time.monotonic()-start
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(output, indent=2))
        print(key, 'random', rows['random']['wins'], 'solver', rows['solver']['wins'], 'auto', rows['solver']['automatic_wins'], flush=True)
    Path(args.replay_output).write_text(json.dumps(episode('solver', args.seed_start+400_000, 16, 40, args.exact_limit, True), indent=2))
    print('Elapsed seconds:', time.monotonic()-start, flush=True)


if __name__ == '__main__':
    main()
