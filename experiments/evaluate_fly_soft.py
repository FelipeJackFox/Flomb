"""Formal evaluation of non-lethal training on raw senses. See research/PROTOCOLO_ENTRENAMIENTO_NO_LETAL.md."""
from pathlib import Path
from experiments import evaluate_fly_raw as base
base.OUT=Path('runs/fly-raw-002');base.GAMES=20000;base.GROUPS=('raw-real-soft','raw-real','table-soft','additive-soft')
if __name__=='__main__':base.main()
