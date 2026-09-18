"""Two fresh reader initializations for the1-vs3-cycle diagnostic."""
from pathlib import Path
import shutil
from experiments import compare_propagation_time as training
from experiments.capacity_probe import write_json
from experiments.train_joint_interface import digest

OUT=Path('runs/propagation-time-consistency-001');SEEDS=(20261104,20261105)
def origin(seed):return Path(f'runs/propagation-time-{seed}')

def main():
    OUT.mkdir(exist_ok=False)
    paths=[training.PARENT,training.MAPPING,Path('runs/hybrid-001/checkpoint.pkl')];hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(seeds=SEEDS,updates=3000,hashes=hashes,primary='mean9 early-late two new reader seeds',secondary='7',tests='750 disjoint fresh layouts per seed',scope='one fixed brain/encoder pretrained at3cycles'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/compare_propagation_time.py'),Path('experiments/report_propagation_time.py'),Path('experiments/report_propagation_time_consistency.py')]:shutil.copy2(p,OUT/'source'/p.name)
    for seed in SEEDS:
        write_json(OUT/'progress.json',dict(phase='replication',seed=seed,detail_file=str(origin(seed)/'progress.json')))
        training.main(out=origin(seed),seed=seed)
    assert all(digest(p)==sha for p,sha in hashes.items())
    write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))

if __name__=='__main__':main()
