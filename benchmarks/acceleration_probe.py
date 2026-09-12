"""Bounded end-to-end comparison using a copied checkpoint and schedule state."""
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from curriculum import draw_board
from minesweeper import Minesweeper
from train_curriculum import train_episode


def main():
    source = Path('runs/curriculum-001')
    out = Path('benchmarks/acceleration')
    out.mkdir(exist_ok=True)
    state = json.loads((source/'state.json').read_text())
    checkpoint = out/'input-checkpoint.npz'
    shutil.copy2(source/state['checkpoint'], checkpoint)
    (out/'input-state.json').write_text(json.dumps(state, indent=2))
    graph = sparse.load_npz('data/processed/graph.npz')
    policy = BrainPolicy(graph)
    results, final = [], None
    for workers, cap in [(1, 0), (1, 1024), (4, 1024)]:
        policy.configure_compute(workers)
        policy.load(checkpoint)
        rng = np.random.default_rng()
        rng.bit_generator.state = state['rng']
        baseline = state['baseline'].copy()
        rows = []
        start = time.perf_counter()
        for episode in range(state['episode'], state['episode']+16):
            group, size, mines, _ = draw_board(rng, episode, state['target'])
            seed = int(rng.integers(0, 1_000_000_000))
            env = Minesweeper(seed, size, mines)
            ret, norm, steps = train_episode(policy, env, rng, baseline[group], cap)
            if steps:
                baseline[group] = .98*baseline[group]+.02*ret
            rows.append([size, mines, seed, ret, norm, steps, bool(env.won)])
        elapsed = time.perf_counter()-start
        current = [a.copy() for a in policy.parameters()+policy.m+policy.v]
        if final is not None:
            assert reference_rows == rows
            assert reference_rng == rng.bit_generator.state
            for a, b in zip(final, current):
                np.testing.assert_array_equal(a, b)
        else:
            final, reference_rows, reference_rng = current, rows, rng.bit_generator.state
        result = dict(workers=workers, cache_mib=cap, episodes=len(rows), clicks=sum(r[5] for r in rows),
                      seconds=elapsed, peak_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      rows=rows, numerical_parity='bitwise exact parameters, moments, episode outcomes and RNG')
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
    policy.close()
    payload = dict(source_checkpoint=state['checkpoint'], source_episode=state['episode'],
                   input_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   note='Measured alongside live training. Timings include environment, forward/backward and Adam; exclude startup/checkpoint I/O. RSS is process lifetime peak.',
                   results=results, end_to_end_speedup=results[0]['seconds']/results[-1]['seconds'])
    (out/'result.json').write_text(json.dumps(payload, indent=2))
    print('SPEEDUP', payload['end_to_end_speedup'], flush=True)


if __name__ == '__main__':
    main()
