"""Balanced, held-out architecture diagnostic; never touches the main run.

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.capacity_probe --out runs/capacity-diagnostic-001
"""
import argparse
import hashlib
import json
import pickle
import resource
import time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.backbone import Backbone, encode_visible
from experiments.expressive_models import ConnectomePolicy, ConvPolicy, equivalent_loss
from experiments.spatial_probe import make_data
from experiments.train import public_context, visible
from minesweeper import Minesweeper


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def category(row):
    """Direct zero/full closure, including public mine total; no subset reasoning.

    Other certified-safe positions require deductions beyond this simple closure.
    This labels the position, not an asserted unique reasoning trace of the agent.
    """
    size = row['size']
    board = row['board'].reshape(16, 16)[:size, :size]
    hidden = set(map(tuple, np.argwhere(board == -1)))
    constraints = [(hidden, row['mines'])]
    for r, c in np.argwhere(board >= 0):
        neighbors = {(rr, cc) for rr in range(max(0, r-1), min(size, r+2))
                     for cc in range(max(0, c-1), min(size, c+2))} & hidden
        constraints.append((neighbors, int(board[r, c])))
    safe, mines = set(), set()
    for _ in range(size*size):
        old = len(safe)+len(mines)
        for cells, total in constraints:
            remaining = cells-safe-mines
            total -= len(cells & mines)
            if total == 0:
                safe |= remaining
            elif total == len(remaining):
                mines |= remaining
        if len(safe)+len(mines) == old:
            break
    return 'elementary' if safe else 'relational_or_exact'


def build_dataset(train_per_size=256, holdout_per_size=64, games_per_size=24):
    used = set()
    train = make_data(3_600_000_000, train_per_size, used)
    holdout = make_data(3_700_000_000, holdout_per_size, used)
    signatures = {r['board'].tobytes() for r in train}
    holdout = [r for r in holdout if r['board'].tobytes() not in signatures]
    for rows in (train, holdout):
        for row in rows:
            row['category'] = category(row)
    games = []
    for size, mines in [(5, 3), (7, 7)]:
        seed = 3_800_000_000 + size*100000
        for _ in range(games_per_size):
            while True:
                env = Minesweeper(seed, size, mines)
                seed += 1
                identity = f'{size}:'+hashlib.sha256(env._mines.tobytes()).hexdigest()
                if identity not in used:
                    used.add(identity)
                    break
            games.append(dict(seed=seed-1, size=size, mines=mines, layout_hash=identity))
    return dict(train=train, holdout=holdout, games=games)


def arrays(rows):
    x, legal = encode_visible([r['board'] for r in rows])
    return (torch.from_numpy(x), torch.from_numpy(legal),
            torch.tensor(np.array([public_context(r['size'], r['mines']) for r in rows])),
            torch.tensor(np.array([r['labels'] for r in rows])))


@torch.no_grad()
def measure(model, rows, batch=16):
    model.eval()
    hits, masses, losses, chance = [], [], [], []
    for i in range(0, len(rows), batch):
        x, legal, context, labels = arrays(rows[i:i+batch])
        z = model(x, context)
        p = z.masked_fill(~legal, -1e9).softmax(1)
        a = p.argmax(1)
        hits.extend(labels[torch.arange(len(a)), a].int().tolist())
        masses.extend((p*labels).sum(1).tolist())
        chance.extend((labels.sum(1)/legal.sum(1)).tolist())
        losses.append(float(equivalent_loss(z, legal, labels))*len(a))
    return dict(correct=sum(hits), n=len(rows), hits=hits,
                safe_probability=float(np.mean(masses)), random_safe_probability=float(np.mean(chance)),
                loss=sum(losses)/len(rows),
                strata={f'{size}/{cat}': dict(correct=sum(h for h, r in zip(hits, rows)
                            if r['size'] == size and r['category'] == cat),
                         n=sum(r['size'] == size and r['category'] == cat for r in rows))
                        for size in (5, 7) for cat in ('elementary', 'relational_or_exact')})


@torch.no_grad()
def play(model, games):
    results = []
    for game in games:
        env = Minesweeper(game['seed'], game['size'], game['mines'])
        automatic, clicks = bool(env.won), 0
        while not env.done:
            x, legal = encode_visible([visible(env)])
            context = torch.from_numpy(public_context(env.size, env.mine_count)[None])
            z = model(torch.from_numpy(x), context).masked_fill(~torch.from_numpy(legal), -1e9)
            a = int(z.argmax(1))
            env.step((a//16)*env.size+a%16)
            clicks += 1
        results.append({**game, 'automatic': automatic, 'won': bool(env.won), 'clicks': clicks})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260922, 20260923])
    parser.add_argument('--variants', nargs='+', choices=['current', 'expressive', 'cnn'], default=['cnn', 'current', 'expressive'])
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out/'dataset.pkl'
    if not dataset_path.exists():
        data = build_dataset()
        with dataset_path.open('wb') as f:
            pickle.dump(data, f)
    data = pickle.loads(dataset_path.read_bytes())
    if args.data_only:
        print('DATASET', len(data['train']), len(data['holdout']), flush=True)
        return
    if (args.out/'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite probe results')
    protected = Path('runs/hybrid-001/checkpoint.pkl')
    protected_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
    sources = ['experiments/capacity_probe.py', 'experiments/expressive_models.py',
               'experiments/spatial_probe.py', 'experiments/backbone.py', 'experiments/train.py',
               'minesweeper.py', 'solver.py']
    (args.out/'source').mkdir()
    for name in sources:
        (args.out/'source'/Path(name).name).write_bytes(Path(name).read_bytes())
    manifest = dict(updates=args.updates, seeds=args.seeds, variants=args.variants, batch=16,
                    lr=.001, torch=torch.__version__, protected_checkpoint_sha256=protected_hash,
                    dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                    source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
                    notes='Fresh parameters; common fixed maps. Current has equivalent mean-Q linear policy, not QR optimizer. Equal updates/batches, not equal compute or parameter count. Architecture bundle screening, not single-factor causal proof.',
                    strata={split: {f'{size}/{cat}':sum(r['size']==size and r['category']==cat for r in data[split])
                        for size in (5,7) for cat in ('elementary','relational_or_exact')} for split in ('train','holdout')})
    write_json(args.out/'manifest.json', manifest)
    operator = Backbone(sparse.load_npz('data/processed/graph.npz'),
                        'runs/curriculum-001/initial-expanded.npz', workers=4)
    groups = {}
    for i, row in enumerate(data['train']):
        groups.setdefault((row['size'], row['category']), []).append(i)
    keys = sorted(groups)
    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        batches = [[int(rng.choice(groups[keys[(step*16+j)%len(keys)]])) for j in range(16)]
                   for step in range(args.updates)]
        for variant in args.variants:
            torch.manual_seed(seed)
            model = ConvPolicy() if variant == 'cnn' else ConnectomePolicy(operator, variant == 'expressive')
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            initial = {n:p.detach().clone() for n,p in model.named_parameters()}
            result = dict(seed=seed, variant=variant, parameters=sum(p.numel() for p in model.parameters()),
                          before=measure(model, data['holdout']), history=[])
            start = time.monotonic()
            for step, indices in enumerate(batches, 1):
                model.train()
                x, legal, context, labels = arrays([data['train'][i] for i in indices])
                optimizer.zero_grad(set_to_none=True)
                loss = equivalent_loss(model(x, context), legal, labels)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if isinstance(model, ConnectomePolicy):
                    with torch.no_grad():
                        model.log_gain.clamp_(-1, 1)
                if step % 30 == 0 or step == args.updates:
                    progress = dict(seed=seed, variant=variant, updates=step, loss=float(loss.detach()),
                                    gradient_norm=float(norm), seconds=time.monotonic()-start)
                    result['history'].append(progress)
                    write_json(args.out/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            result['training_seconds'] = time.monotonic()-start
            result['changed_parameters'] = {n:int((p.detach()!=initial[n]).sum()) for n,p in model.named_parameters()}
            result['train'] = measure(model, data['train'])
            result['holdout'] = measure(model, data['holdout'])
            result['games'] = play(model, data['games'])
            result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # The peak is cumulative for the process, NOT a per-model allocation measurement.
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), seed=seed,
                            updates=args.updates, variant=variant, torch_rng=torch.get_rng_state(),
                            dataset_sha256=manifest['dataset_sha256']), args.out/f'{variant}-{seed}.pt')
            results.append(result)
            write_json(args.out/'results.json', results)
            print('FINISHED', variant, seed, result['holdout']['correct'], flush=True)
            del model, optimizer, initial
    operator.close()
    intact = hashlib.sha256(protected.read_bytes()).hexdigest() == protected_hash
    if not intact:
        raise RuntimeError('Protected hybrid checkpoint changed during probe')
    write_json(args.out/'completed.json', dict(time=time.time(), models=len(results), protected_checkpoint_intact=intact))


if __name__ == '__main__':
    main()
