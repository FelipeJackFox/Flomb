"""Front 3b, first piece: real APL feedback inhibition (per-Kenyon-cell synapse counts) instead of an exact top-k cut."""
from experiments import evaluate_fly_wiring as W
ARMS={'apl-real':dict(inhibition='apl'),'topk-real':dict(),'apl-shuffled':dict(inhibition='apl',shuffled=True),'apl-random':dict(inhibition='apl',shuffled='random')}
if __name__=='__main__':W.main(out='runs/fly-apl-001',arms=ARMS,baseline='apl-real')
