"""Visible-only teacher and random references on the exact probe game boards."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from minesweeper import Minesweeper
from solver import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    data = pickle.loads((args.run/'dataset.pkl').read_bytes())
    results = {}
    for variant in ('random', 'visible_teacher'):
        rows = []
        for game in data['games']:
            env = Minesweeper(game['seed'], game['size'], game['mines'])
            rng = np.random.default_rng(game['seed']+100)
            automatic = bool(env.won)
            while not env.done:
                if variant == 'random':
                    candidates = np.flatnonzero(env.legal_mask())
                else:
                    info = analyze(env.observation(), env.legal_mask(), env.size, env.mine_count)
                    candidates = sorted(info.safe)
                    if not candidates:
                        legal = [int(i) for i in np.flatnonzero(env.legal_mask()) if i not in info.mines]
                        risk = min(info.mine_probability[i] for i in legal)
                        candidates = [i for i in legal if info.mine_probability[i] <= risk+1e-12]
                env.step(int(rng.choice(candidates)))
            rows.append({**game, 'automatic': automatic, 'won': bool(env.won)})
        results[variant] = rows
    (args.run/'references.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
