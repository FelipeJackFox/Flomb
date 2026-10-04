"""Large replication of the working configuration. See research/PROTOCOLO_REPLICA_NUMERO_DURO.md."""
from pathlib import Path
from experiments import evaluate_fly_aggregate as A
A.OUT=Path('runs/fly-replica-001');A.SEEDS=tuple(range(20,50));A.ARMS={'meanown-cap':('meanown',0.)};A.SETS=((9,12,2000),(7,7,500),(16,40,300))
if __name__=='__main__':A.main()
