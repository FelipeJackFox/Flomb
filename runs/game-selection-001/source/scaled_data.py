"""10k distinct visible positions; partition whole mine layouts before collecting states."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from minesweeper import Minesweeper
from solver import analyze
from experiments.train import visible, teacher
from experiments.capacity_probe import category, write_json


def identity(env):
    # Hidden truth is used exclusively for split identities, never model/teacher inputs.
    return f'{env.size}:'+hashlib.sha256(env._mines.tobytes()).hexdigest()


def trajectory(env, seed):
    rng = np.random.default_rng(seed+19)
    rows = []
    move = 0
    layout = identity(env)
    while not env.done:
        info = analyze(env.observation(), env.legal_mask(), env.size, env.mine_count)
        labels, _ = teacher(env, info)
        if info.safe:
            row = dict(board=visible(env), labels=labels, size=env.size, mines=env.mine_count,
                       seed=seed, layout_hash=layout, move=move)
            row['category'] = category(row)
            rows.append(row)
        candidates = np.flatnonzero(labels)
        action = int(rng.choice(candidates))
        env.step((action//16)*env.size+action%16)
        move += 1
    rng.shuffle(rows)
    return rows


def build(out, train_per_size=5000, valid_per_size=500, games_per_size=250):
    used, visible_used, previous = set(), set(), set()
    for name in ('runs/capacity-diagnostic-001/dataset.pkl', 'runs/spatial-diagnostic-001/dataset.pkl'):
        path = Path(name)
        if path.exists():
            for rows in pickle.loads(path.read_bytes()).values():
                previous.update(r['layout_hash'] for r in rows)
    data = {'train': [], 'holdout': [], 'games': []}
    started = time.monotonic()
    # Reserve the 500 final game boards FIRST, including layouts with no safe opening.
    for size, mines in ((5, 3), (7, 7)):
        seed = 4_000_000_000+size*100000
        count = 0
        while count < games_per_size:
            env = Minesweeper(seed, size, mines)
            layout = identity(env)
            if layout not in used and layout not in previous:
                data['games'].append(dict(seed=seed, size=size, mines=mines, layout_hash=layout))
                used.add(layout)
                count += 1
            seed += 1
    # Multiple distinct positions may come from one layout, always within one split.
    for split, target, start in (('holdout', valid_per_size, 4_100_000_000),
                                 ('train', train_per_size, 4_200_000_000)):
        for size, mines in ((5, 3), (7, 7)):
            seed = start+size*100000
            count, attempts = 0, 0
            while count < target:
                attempts += 1
                if attempts > 100000:
                    raise RuntimeError(f'Finite layout pool exhausted for {split} {size}: {count}/{target}')
                env = Minesweeper(seed, size, mines)
                layout = identity(env)
                if layout not in used:
                    used.add(layout)
                    for row in trajectory(env, seed):
                        signature = row['board'].tobytes()
                        if signature in visible_used:
                            continue
                        visible_used.add(signature)
                        data[split].append(row)
                        count += 1
                        if count == target:
                            break
                seed += 1
                if attempts % 100 == 0:
                    progress = dict(phase='dataset', split=split, size=size, collected=count,
                                    target=target, layouts_seen=len(used), seconds=time.monotonic()-started)
                    write_json(out/'data-progress.json', progress)
                    print(json.dumps(progress), flush=True)
    for a,b in (('train','holdout'),('train','games'),('holdout','games')):
        assert not {r['layout_hash'] for r in data[a]} & {r['layout_hash'] for r in data[b]}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'dataset.pkl'
    if path.exists():
        raise FileExistsError(path)
    data = build(args.out)
    with path.with_suffix('.tmp').open('wb') as f:
        pickle.dump(data, f)
    path.with_suffix('.tmp').replace(path)
    summary = {split: dict(n=len(rows), unique_layouts=len({r['layout_hash'] for r in rows}),
                   strata={f'{size}/{cat}':sum(r['size']==size and r.get('category')==cat for r in rows)
                           for size in (5,7) for cat in ('elementary','relational_or_exact')})
               for split,rows in data.items()}
    write_json(args.out/'dataset-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
