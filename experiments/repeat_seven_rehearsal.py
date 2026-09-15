"""Two new fixed-budget retention replications, independently reserved tests."""
import json,shutil
from pathlib import Path
from experiments import train_seven_rehearsal as training
from experiments.capacity_probe import write_json
from experiments.train_risk_auxiliary import digest

OUT=Path('runs/seven-rehearsal-consistency-001');SEEDS=(20261003,20261004)
def origin(seed):return Path(f'runs/seven-rehearsal-{seed}')

def main():
    OUT.mkdir(exist_ok=False)
    paths=[Path(f'runs/nine-dagger-{s}/latest-dagger.pt') for s in SEEDS]
    paths += [Path('runs/hybrid-001/checkpoint.pkl')]
    hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(seeds=SEEDS,updates=1500,hashes=hashes,
        primary='mean7 replay minus control across two NEW seeds',secondary='9 cost, vsparent9',
        tests='750 distinct new layouts per seed, disjoint across seeds and history',
        checkpoint='fixed last1500 each; pilot02 excluded from primary'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/train_seven_rehearsal.py'),Path('experiments/report_seven_rehearsal.py'),Path('experiments/report_seven_rehearsal_consistency.py')]:shutil.copy2(p,OUT/'source'/p.name)
    for seed in SEEDS:
        write_json(OUT/'progress.json',dict(phase='replication',seed=seed,detail_file=str(origin(seed)/'progress.json')))
        training.main(out=origin(seed),seed=seed)
    assert all(digest(p)==sha for p,sha in hashes.items())
    write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))

if __name__=='__main__':main()
