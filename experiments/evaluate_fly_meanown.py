"""Formal confirmation of per-sniff conditioning + averaged decision, on seeds never used while designing it."""
from pathlib import Path
from experiments import evaluate_fly_aggregate as A
A.OUT=Path('runs/fly-aggregate-002');A.SEEDS=tuple(range(10,20));A.ARMS={'sum-cap':('sum',0.),'mean-cap':('mean',0.),'meanown-cap':('meanown',0.)}
if __name__=='__main__':A.main()
