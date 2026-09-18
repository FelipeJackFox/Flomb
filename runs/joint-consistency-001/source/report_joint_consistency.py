"""Audit repeated encoder adaptation; primary analysis excludes discovery seed."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.repeat_joint_interface import OUT,SEEDS,origin
from experiments.train_joint_interface import digest,tensor_hash,BASE
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 manifest=json.loads((OUT/'manifest.json').read_text());results=json.loads((OUT/'results.json').read_text());games=json.loads((OUT/'games.json').read_text())
 chosen=json.loads((OUT/'selection.json').read_text());assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
 assert all(digest(r['path'])==r['sha256'] for r in chosen.values())
 keys={r['layout_hash'] for r in games};assert len(keys)==500
 for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 cohort={OUT,*[origin(s) for s in SEEDS[1:]]};used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent in cohort:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/*games.json'):
  if p.parent in cohort:continue
  rows=json.loads(p.read_text())
  if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 assert not used&keys
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);reference=None;wins={};selection={}
 source=torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model'];encoder={k.removeprefix('encoder.'):v for k,v in source.items() if k.startswith('encoder.')}
 weightids={}
 for seed in SEEDS:
  p=origin(seed);pair=json.loads((p/'pairing.json').read_text());assert pair['control']==pair['joint']
  for arm in ('baseline','control','joint'):
   key=f'{seed}-{arm}'
   if arm=='baseline':
    saved=torch.load(f'runs/input-representation-001/best-brain-{seed}.pt',weights_only=False);state=encoder;step=0
   else:
    saved=torch.load(p/f'best-{arm}.pt',weights_only=False);state=saved['encoder'];step=saved['step']
    hist=json.loads((p/f'history-{arm}.json').read_text());assert [r['step'] for r in hist]==[0,250,500,750]
    best=min(hist,key=lambda r:r['validation_loss']);assert step==best['step'] and saved['validation_loss']==best['validation_loss']
   selection[key]=step;weightids[key]=tensor_hash(state)+tensor_hash(saved['head'])
   rr=json.loads((OUT/f'{key}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
   auto=[r['automatic'] for r in rr]
   if reference is None:reference=auto
   assert reference==auto;played=[r for r in rr if not r['automatic']]
   wins[key]=np.array([int(r['won']) for r in played]);assert int(wins[key].sum())==results[key]['wins'] and len(played)==results[key]['n']
   for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==results[key][metric]
   if 'duplicate_of' in results[key]:assert weightids[key]==weightids[results[key]['duplicate_of']]
 for seed in SEEDS[1:]:
  p=origin(seed);m=json.loads((p/'manifest.json').read_text());assert all(digest(path)==sha for path,sha in m['hashes'].items())
  assert all(json.loads((p/'training-verification.json').read_text()).values())
  assert json.loads((p/'games.json').read_text())==games
 comparisons={}
 for name,seeds in [('new_seeds',SEEDS[1:]),('all_seeds',SEEDS),*[(str(s),(s,)) for s in SEEDS]]:
  delta=np.stack([wins[f'{s}-joint']-wins[f'{s}-control'] for s in seeds]).mean(0)
  rng=np.random.default_rng(20261020);boot=[rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)]
  comparisons[name]=dict(control_mean_pct=float(np.mean([wins[f'{s}-control'].mean() for s in seeds])*100),joint_mean_pct=float(np.mean([wins[f'{s}-joint'].mean() for s in seeds])*100),delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(comparisons=comparisons,results=results,selected_steps=selection,shared_layouts=500,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary)
 write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=500,prior_overlap=0,all_selections_recomputed=True,duplicate_weights_verified=True,all_original_hashes_intact=True,selection_seal_intact=True,paired_draws_verified=True,result_counts_verified=True))
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');xx=np.arange(3)
 for j,(arm,label,color) in enumerate([('control','Lector solo','#a768af'),('joint','Entrada + lector','#078d9d')]):
  vals=[100*wins[f'{s}-{arm}'].mean() for s in SEEDS];x=xx+(j-.5)*.34;axes[0].bar(x,vals,.34,label=label,color=color)
  for xpos,v,s in zip(x,vals,SEEDS):axes[0].text(xpos,v+.5,str(results[f'{s}-{arm}']['wins']),ha='center')
 axes[0].set_xticks(xx,['20261002\nPiloto reutilizado','20261003\nRepetición nueva','20261004\nRepetición nueva']);axes[0].set_ylabel('Victorias autónomas (%)');axes[0].set_ylim(0,max(100*w.mean() for w in wins.values())+7);axes[0].legend();axes[0].set_title('Mismos 500 tableros nuevos · 7×7 / 7 minas')
 for i,group in enumerate(('new_seeds','all_seeds')):
  r=comparisons[group];lo,hi=r['conditional_ci95_pp'];axes[1].errorbar(r['delta_pp'],i,xerr=[[r['delta_pp']-lo],[hi-r['delta_pp']]],fmt='o',capsize=5,color='#078d9d')
 axes[1].axvline(0,color='#888',linestyle='--');axes[1].set_yticks([0,1],['Dos nuevas: principal','Tres: secundario']);axes[1].set_xlabel('Diferencia de victorias (puntos porcentuales)');axes[1].set_title('Intervalos pareados del 95%')
 fig.savefig('research/joint-consistency-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {s} | {results[f'{s}-baseline']['wins']}/{results[f'{s}-baseline']['n']} | {results[f'{s}-control']['wins']}/{results[f'{s}-control']['n']} | {results[f'{s}-joint']['wins']}/{results[f'{s}-joint']['n']} | {comparisons[str(s)]['delta_pp']:+.1f} |" for s in SEEDS)
 main=comparisons['new_seeds'];lo,hi=main['conditional_ci95_pp']
 doc=f'''# Consistencia del entrenamiento de entrada y lector

Comparación principal de las DOS semillas nuevas: **{main['control_mean_pct']:.2f}%→{main['joint_mean_pct']:.2f}%**, diferencia **{main['delta_pp']:+.2f} puntos**, IC95% condicional [{lo:+.2f},{hi:+.2f}]. El piloto reutilizado se excluye del análisis principal porque su resultado motivó la repetición.

| Semilla | Inicial | Control | Conjunto | Diferencia vs control (pp) |
|---|---|---|---|---|
{table}

![Consistencia](joint-consistency-results.png)

Dos entrenamientos nuevos con750updates por brazo, mismos datos/Adam/RNG iniciales dentro de cada par, batch64/micro16, mismas tasas y clipping del piloto. Cada semilla parte de su checkpoint cerebral previo. El piloto20261002 no se reentrenó. Selección original holdout0/250/500/750 sellada para los seis candidatos antes de probar500layouts nuevos compartidos7×7/7minas. {sum(reference)} aperturas ganadoras automáticas excluidas. Grafo y ganancias fijos. Sin DAgger adicional ni reemplazo del agente servido.

Intervalos de10000 remuestreos por tablero, promediando las diferencias entre semillas dentro del tablero. Condicionados a estos modelos; un único cerebro,500tableros compartidos, no1500 observaciones independientes. Tres semillas de lector no establecen universalidad ni ventaja anatómica. Mismo número de ejemplos/actualizaciones, distinto tiempo y parámetros entrenables entre control/conjunto. No comparar directamente con el~20% de modelos DAgger en otros benchmarks.

```json
{json.dumps(summary,indent=2)}
```

Auditoría independiente:500layouts reconstruidos,solapamiento previo0,reglas de selección recalculadas,minibatches pareados,conteos,hases y sello correctos. Duplicados de evaluación solo por encoder+lector idénticos. Cachés de actividad separados por encoder. Pruebas gradientes/forward del protocolo conservadas. Fuentes y checkpoints en `runs/joint-consistency-001` y sus dos corridas de entrenamiento. Protocolo `research/PROTOCOLO_CONSISTENCIA_INTERFAZ.md`.
'''
 Path('research/RESULTADO_CONSISTENCIA_INTERFAZ.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
