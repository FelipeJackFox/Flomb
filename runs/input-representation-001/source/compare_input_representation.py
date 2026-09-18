"""Matched spatial readers: public clue maps versus frozen connectome activity."""
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from experiments.capacity_probe import arrays, write_json
from experiments.extend_spatial_decoder import BASE, CACHE, digest, validate
from experiments.expressive_models import equivalent_loss
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.scaled_data import identity
from experiments.scaled_train import evaluate_games
from experiments.spatial_decoder import ActivityHead, activity_map
from experiments.validate_spatial_decoder import ActivityMemo, MemoPolicy, SEEDS
from minesweeper import Minesweeper

OUT = Path('runs/input-representation-001')
UPDATES = 5000

def raw_map(x):
    return x.reshape(-1, 16, 16, 10).permute(0, 3, 1, 2).contiguous()

class RawPolicy(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, context):
        return self.head(raw_map(x), context)

def reserve():
    used = set()
    for path in Path('runs').glob('*/dataset.pkl'):
        for rows in pickle.loads(path.read_bytes()).values():
            if isinstance(rows, list):
                used.update(r['layout_hash'] for r in rows if isinstance(r, dict) and 'layout_hash' in r)
    for path in Path('runs').glob('*/*games.json'):
        rows = json.loads(path.read_text())
        if isinstance(rows, list):
            used.update(r['layout_hash'] for r in rows if isinstance(r, dict) and 'layout_hash' in r)
    games = []
    for seed in range(6800700000, 6800800000):
        key = identity(Minesweeper(seed, 7, 7))
        if key not in used:
            games.append(dict(seed=seed, size=7, mines=7, layout_hash=key))
            used.add(key)
        if len(games) == 500:
            return games
    raise RuntimeError('Fresh layout search exhausted')

def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    paths = [BASE/'dataset.pkl', BASE/'mapping.pkl', BASE/'training/retina_plastic-20260926.pt',
             CACHE/'train.pt', CACHE/'holdout.pt', Path('runs/hybrid-001/checkpoint.pkl')]
    hashes = {str(p): digest(p) for p in paths}
    manifest = dict(seeds=SEEDS, updates=UPDATES, batch=64, lr=.001, parameters=12769,
                    selection='minimum original holdout loss, step0 and every250; earliest tie',
                    frozen='existing brain and encoder; both heads initialized from scratch',
                    data='original 10000 positions, 1000 holdout; no additional DAgger',
                    pairing='identical initial weights and minibatch indices per seed',
                    scope='conditional on one pretrained frozen brain, not a comparison of total historical training compute',
                    hashes=hashes, torch=torch.__version__)
    write_json(OUT/'manifest.json', manifest)
    source = OUT/'source'; source.mkdir()
    for p in [Path(__file__), *[Path('experiments')/n for n in
            ('spatial_decoder.py', 'expressive_models.py', 'capacity_probe.py', 'scaled_train.py',
             'validate_spatial_decoder.py', 'retina_policy.py', 'plastic_sparse.py', 'backbone.py', 'train.py')],
             Path('solver.py'), Path('minesweeper.py')]:
        shutil.copy2(p, source/p.name)
    games = reserve(); write_json(OUT/'games.json', games)
    with (OUT/'dataset.pkl').open('wb') as f:
        pickle.dump(dict(games=games), f)
    data = pickle.loads((BASE/'dataset.pkl').read_bytes())
    cache = {'raw': {}, 'brain': {}}
    for split in ('train', 'holdout'):
        x, legal, context, labels = arrays(data[split])
        cache['raw'][split] = (raw_map(x), legal, context, labels)
        cache['brain'][split] = torch.load(CACHE/f'{split}.pt', weights_only=False)
        for expected, actual in zip((legal, context, labels), cache['brain'][split][1:]):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert not {r['layout_hash'] for r in data['train']} & {r['layout_hash'] for r in data['holdout']}
    groups = {}
    for i, row in enumerate(data['train']):
        groups.setdefault((row['size'], row['category']), []).append(i)
    groups = [np.array(groups[k]) for k in sorted(groups)]
    selection = {}; training_hashes = {}
    for seed in SEEDS:
        for variant in ('raw', 'brain'):
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            head = ActivityHead(True); opt = torch.optim.Adam(head.parameters(), lr=.001)
            assert sum(p.numel() for p in head.parameters()) == manifest['parameters']
            initial = hashlib.sha256(b''.join(v.numpy().tobytes() for v in head.state_dict().values())).hexdigest()
            batches = hashlib.sha256(); best = float('inf'); history = []; started = time.monotonic()
            for step in range(UPDATES+1):
                if step:
                    ix = np.array([int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(64)], dtype=np.int64)
                    batches.update(ix.tobytes())
                    a, l, c, y = [v[ix] for v in cache[variant]['train']]
                    opt.zero_grad(); loss = equivalent_loss(head(a, c), l, y)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True); opt.step()
                if step % 250 == 0:
                    value = validate(head, cache[variant]['holdout'])
                    state = dict(head=head.state_dict(), optimizer=opt.state_dict(), rng=rng.bit_generator.state,
                                 step=step, validation_loss=value, seed=seed, variant=variant)
                    if value < best:
                        best = value; chosen = step
                        torch.save(state, OUT/f'best-{variant}-{seed}.pt')
                    torch.save(state, OUT/f'latest-{variant}-{seed}.pt')
                    progress = dict(phase='training', variant=variant, seed=seed, step=step,
                                    validation_loss=value, chosen=chosen, seconds=time.monotonic()-started)
                    history.append(progress); write_json(OUT/'progress.json', progress)
                    write_json(OUT/f'history-{variant}-{seed}.json', history); print(progress, flush=True)
            key = f'{variant}-{seed}'
            selection[key] = dict(step=chosen, validation_loss=best)
            training_hashes[key] = dict(initial=initial, batches=batches.hexdigest())
        assert training_hashes[f'raw-{seed}'] == training_hashes[f'brain-{seed}']
    write_json(OUT/'pairing.json', training_hashes)
    write_json(OUT/'selection.json', selection)
    seal = digest(OUT/'selection.json'); write_json(OUT/'selection-seal.json', dict(sha256=seal))
    del cache
    op = PlasticOperator(sparse.load_npz('data/processed/graph.npz'), workers=4)
    brain = RetinaPolicy(op, pickle.loads((BASE/'mapping.pkl').read_bytes()), True)
    brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt', weights_only=False)['model'])
    brain.requires_grad_(False); brain.eval(); memo = ActivityMemo(brain)
    # Recompute separated cache rows against the exact frozen checkpoint.
    for split in ('train', 'holdout'):
        ix = np.linspace(0, len(data[split])-1, 8, dtype=int)
        x, _, c, _ = arrays([data[split][i] for i in ix])
        cached = torch.load(CACHE/f'{split}.pt', weights_only=False)[0][ix]
        torch.testing.assert_close(activity_map(brain, x, c), cached, rtol=1e-5, atol=1e-5)
    results = {}
    for seed in SEEDS:
        for variant in ('raw', 'brain'):
            key = f'{variant}-{seed}'; head = ActivityHead(True)
            head.load_state_dict(torch.load(OUT/f'best-{key}.pt', weights_only=False)['head'])
            model = RawPolicy(head) if variant == 'raw' else MemoPolicy(memo, head)
            write_json(OUT/'progress.json', dict(phase='evaluation', variant=key, games=500))
            started = time.monotonic(); rows = evaluate_games(model, games)
            write_json(OUT/f'{key}-games.json', rows)
            played = [r for r in rows if not r['automatic']]
            results[key] = dict(wins=sum(r['won'] for r in played), n=len(played),
                                automatic_excluded=len(rows)-len(played), seconds=time.monotonic()-started,
                                known_mine_choices=sum(r['known_mine_choices'] for r in played),
                                safe_choices=sum(r['safe_choices'] for r in played),
                                safe_opportunities=sum(r['safe_opportunities'] for r in played))
            write_json(OUT/'results.json', results); print(key, results[key], flush=True)
    assert digest(OUT/'selection.json') == seal
    assert all(digest(p) == sha for p, sha in hashes.items())
    write_json(OUT/'verification.json', dict(original_hashes_unchanged=True, paired_initialization_and_batches=True,
               cache_metadata_exact=True, cache_recomputed_samples=16, selection_sealed=True))
    write_json(OUT/'completed.json', dict(completed=True))
    write_json(OUT/'progress.json', dict(phase='completed')); op.close()

if __name__ == '__main__':
    main()
