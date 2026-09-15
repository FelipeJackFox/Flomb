"""Two fresh reader initializations for the1-vs3-cycle diagnostic."""
from pathlib import Path
import shutil
from experiments import evaluate_rotation_readout as training
from experiments.capacity_probe import write_json
from experiments.train_joint_interface import digest

OUT=Path('runs/rotation-consistency-001');SEEDS=(20261104,20261105)
def origin(seed):return Path(f'runs/rotation-readout-{seed}')

def main():
    OUT.mkdir(exist_ok=False)
    paths=[training.PARENT,training.MAPPING,Path('runs/hybrid-001/checkpoint.pkl')];hashes={str(p):digest(p) for p in paths}
    write_json(OUT/'manifest.json',dict(seeds=SEEDS,updates=0,hashes=hashes,primary='mean9 rotated-identity two other adapted models',secondary='7',tests='750 disjoint fresh layouts per seed',scope='fixed models,1cycle,4orientations vs identity,0updates'))
    (OUT/'source').mkdir()
    for p in [Path(__file__),Path('experiments/evaluate_rotation_readout.py'),Path('experiments/report_rotation_readout.py'),Path('experiments/report_rotation_consistency.py')]:shutil.copy2(p,OUT/'source'/p.name)
    for seed in SEEDS:
        write_json(OUT/'progress.json',dict(phase='replication',seed=seed,detail_file=str(origin(seed)/'progress.json')))
        training.main(out=origin(seed),parent_path=Path(f'runs/early-interface-{seed}/latest-joint.pt'))
    assert all(digest(p)==sha for p,sha in hashes.items())
    write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))

if __name__=='__main__':main()
