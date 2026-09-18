"""Fixed policies across new board sizes; diagnostic, no checkpoint selection."""
import json
import pickle
import shutil
from pathlib import Path

import torch
from scipy import sparse

from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments.scaled_train import evaluate_games
from experiments.spatial_decoder import ActivityHead
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.validate_spatial_decoder import ActivityMemo, MemoPolicy
from experiments.train_risk_auxiliary import BASE, digest, tensor_hash
from minesweeper import Minesweeper

OUT = Path('runs/difficulty-transfer-001')
SEEDS = (20261002, 20261003, 20261004)
LEVELS = ((7, 7), (9, 12), (12, 22), (16, 38))
N = 200


def prior_layouts(exclude=OUT):
    used = set()
    for path in Path('runs').glob('*/dataset.pkl'):
        if path.parent == exclude:
            continue
        for rows in pickle.loads(path.read_bytes()).values():
            if isinstance(rows, list):
                used.update(r['layout_hash'] for r in rows if isinstance(r, dict) and 'layout_hash' in r)
    for path in Path('runs').glob('*/*games.json'):
        if path.parent != exclude:
            rows = json.loads(path.read_text())
            if isinstance(rows, list):
                used.update(r['layout_hash'] for r in rows if isinstance(r, dict) and 'layout_hash' in r)
    return used


def reserve(out=OUT):
    used, games = prior_layouts(exclude=out), []
    for size, mines in LEVELS:
        found = 0
        for seed in range(8100000000 + size * 100000, 8100000000 + (size + 1) * 100000):
            key = identity(Minesweeper(seed, size, mines))
            if key in used:
                continue
            used.add(key)
            games.append(dict(seed=seed, size=size, mines=mines, layout_hash=key))
            found += 1
            if found == N:
                break
        assert found == N
    return games


def main(out=OUT, expanded=False):
    global OUT
    OUT = Path(out)
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    parents = {s: Path(f'runs/wide-reader-001/best-{s}-local.pt') for s in SEEDS}
    paths = [*parents.values(), BASE/'training/retina_plastic-20260926.pt',
             BASE/'mapping.pkl', Path('runs/hybrid-001/checkpoint.pkl')]
    hashes = {str(p): digest(p) for p in paths}
    write_json(OUT/'manifest.json', dict(training=False, seeds=SEEDS, levels=LEVELS,
        games_per_level=N, hashes=hashes, expanded_mapping=expanded,
        selection='fixed preserved local heads; no test selection',
        scope='three policies, one brain; 800 shared new layouts, roughly 15% mines',
        primary='autonomous win rate by size',
        secondary='safe choices/opportunities, deaths with safe available, certified mine choices, exact half-risk deaths',
        caution='solver-certified safe is not all possible logical safety; no luck-adjusted wins'))
    src = OUT/'source'
    src.mkdir()
    for p in [Path(__file__), Path('experiments/report_difficulty_transfer.py'),
              Path('experiments/scaled_train.py'), Path('experiments/spatial_decoder.py'),
              Path('experiments/retina_policy.py'), Path('solver.py'), Path('minesweeper.py')]:
        shutil.copy2(p, src/p.name)
    games = reserve(OUT)
    write_json(OUT/'games.json', games)
    with (OUT/'dataset.pkl').open('wb') as f:
        pickle.dump(dict(games=games), f)
    op = PlasticOperator(sparse.load_npz('data/processed/graph.npz'), workers=4)
    try:
        brain = RetinaPolicy(op, pickle.loads((BASE/'mapping.pkl').read_bytes()), True)
        brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt', weights_only=False)['model'])
        if expanded:
            from experiments.retina_policy import build_mapping
            mapping = build_mapping(sizes=(5, 7, 9, 12, 16))
            for size, (ix, wt) in mapping['outputs'].items():
                if hasattr(brain, f'out_{size}'):
                    assert torch.equal(getattr(brain, f'out_{size}'), torch.from_numpy(ix))
                    assert torch.equal(getattr(brain, f'weight_{size}'), torch.from_numpy(wt))
                else:
                    brain.register_buffer(f'out_{size}', torch.from_numpy(ix), persistent=False)
                    brain.register_buffer(f'weight_{size}', torch.from_numpy(wt), persistent=False)
            with (OUT/'extended-mapping.pkl').open('wb') as f:
                pickle.dump(mapping, f)
            write_json(OUT/'mapping-verification.json', dict(old_5_7_exact=True,
                mapping_sha256=digest(OUT/'extended-mapping.pkl'), new_sizes=[9,12,16]))
        brain.requires_grad_(False)
        for seed in SEEDS:
            saved = torch.load(parents[seed], weights_only=False)
            brain.encoder.load_state_dict(saved['encoder'])
            frozen = tensor_hash(brain.state_dict())
            head = ActivityHead(True)
            head.load_state_dict(saved['head'])
            for size, mines in LEVELS:
                rows = []
                selected = [g for g in games if g['size'] == size]
                # Each batch gets fresh cache: bound memory and never mix encoders.
                for start in range(0, N, 50):
                    policy = MemoPolicy(ActivityMemo(brain), head)
                    rows.extend(evaluate_games(policy, selected[start:start+50]))
                    write_json(OUT/f'{seed}-{size}-games.json', rows)
                    record = dict(phase='evaluation', seed=seed, size=size, mines=mines,
                                  completed=len(rows), total=N)
                    write_json(OUT/'progress.json', record)
                    print(record, flush=True)
                assert tensor_hash(brain.state_dict()) == frozen
        assert all(digest(p) == sha for p, sha in hashes.items())
        write_json(OUT/'verification.json', dict(originals_intact=True, brain_frozen=True, no_training=True))
        write_json(OUT/'completed.json', dict(completed=True))
        write_json(OUT/'progress.json', dict(phase='completed'))
    finally:
        op.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--expand-mapping', action='store_true')
    args = parser.parse_args()
    main(args.out, args.expand_mapping)
