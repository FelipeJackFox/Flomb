"""Audit the resumed experiment, including preserved750 models on fresh games."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.extend_joint_interface import OUT,SEEDS,origin,previous
from experiments.train_joint_interface import digest,tensor_hash,BASE
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 manifest=json.loads((OUT/'manifest.json').read_text());results=json.loads((OUT/'results.json').read_text());games=json.loads((OUT/'games.json').read_text());chosen=json.loads((OUT/'selection.json').read_text())
 assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 assert all(digest(p)==sha for p,sha in manifest['hashes'].items()) and all(digest(r['path'])==r['sha256'] for r in chosen.values())
 keys={r['layout_hash'] for r in games};assert len(keys)==500
 for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 cohort={OUT,*[origin(s) for s in SEEDS]};used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent in cohort:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/*games.json'):
  if p.parent in cohort:continue
  rows=json.loads(p.read_text())
  if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 assert not keys&used
 source=torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model'];encoder={k.removeprefix('encoder.'):v for k,v in source.items() if k.startswith('encoder.')}
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);reference=None;wins={};weightids={};steps={}
 for seed in SEEDS:
  p=origin(seed);pair=json.loads((p/'pairing.json').read_text());assert pair['control']==pair['joint']
  assert all(json.loads((p/'training-verification.json').read_text()).values())
  m=json.loads((p/'manifest.json').read_text());assert all(digest(path)==sha for path,sha in m['hashes'].items())
  assert json.loads((p/'games.json').read_text())==games
  for arm in ('baseline','control','joint','prior'):
   key=f'{seed}-{arm}'
   if arm=='baseline':saved=torch.load(f'runs/input-representation-001/best-brain-{seed}.pt',weights_only=False);state=encoder;step=0
   else:
    file=previous(seed)/'best-joint.pt' if arm=='prior' else p/f'best-{arm}.pt';saved=torch.load(file,weights_only=False);state=saved['encoder'];step=saved['step']
    if arm!='prior':
     hist=json.loads((p/f'history-{arm}.json').read_text());assert [r['step'] for r in hist]==list(range(0,3001,250))
     best=min(hist,key=lambda r:r['validation_loss']);assert step==best['step'] and saved['validation_loss']==best['validation_loss']
     assert hist[:4]==json.loads((previous(seed)/f'history-{arm}.json').read_text())
     resume=json.loads((p/f'resume-{arm}.json').read_text());assert resume['source_step']==750 and resume['weights_optimizer_rng_exact']
     if arm=='control':assert tensor_hash(state)==tensor_hash(encoder)
   steps[key]=step;weightids[key]=tensor_hash(state)+tensor_hash(saved['head'])
   rr=json.loads((OUT/f'{key}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
   auto=[r['automatic'] for r in rr]
   if reference is None:reference=auto
   assert reference==auto;played=[r for r in rr if not r['automatic']]
   wins[key]=np.array([int(r['won']) for r in played]);assert int(wins[key].sum())==results[key]['wins'] and len(played)==results[key]['n']
   for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==results[key][metric]
   if 'duplicate_of' in results[key]:assert weightids[key]==weightids[results[key]['duplicate_of']]
 comparisons={}
 for ref in ('prior','control'):
  delta=np.stack([wins[f'{s}-joint']-wins[f'{s}-{ref}'] for s in SEEDS]).mean(0);rng=np.random.default_rng(20261021);boot=[rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)]
  comparisons[ref]=dict(reference_mean_pct=float(np.mean([wins[f'{s}-{ref}'].mean() for s in SEEDS])*100),joint_mean_pct=float(np.mean([wins[f'{s}-joint'].mean() for s in SEEDS])*100),delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(comparisons=comparisons,results=results,selected_steps=steps,shared_layouts=500,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary)
 write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=500,prior_overlap=0,all_selections_recomputed=True,resume_records_and_history_verified=True,paired_draws_verified=True,duplicates_verified=True,hashes_and_seal_intact=True,result_counts_verified=True))
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(3)
 for j,(arm,label,color) in enumerate([('prior','Conjunto750','#909ca2'),('control','Control ampliado','#a768af'),('joint','Conjunto ampliado','#078d9d')]):
  vals=[100*wins[f'{s}-{arm}'].mean() for s in SEEDS];xx=x+(j-1)*.24;axes[0].bar(xx,vals,.24,label=label,color=color)
  for xpos,v,s in zip(xx,vals,SEEDS):axes[0].text(xpos,v+.5,str(results[f'{s}-{arm}']['wins']),ha='center',fontsize=9)
 axes[0].set_xticks(x,[str(s) for s in SEEDS]);axes[0].set_ylabel('Victorias autónomas (%)');axes[0].set_ylim(0,max(100*w.mean() for w in wins.values())+8);axes[0].legend(fontsize=9);axes[0].set_title('500 tableros nuevos compartidos · 7×7 / 7 minas')
 for seed in SEEDS:
  h=json.loads((origin(seed)/'history-joint.json').read_text());axes[1].plot([r['step'] for r in h],[r['validation_loss'] for r in h],label=str(seed))
 axes[1].axvline(750,color='#888',linestyle='--');axes[1].set(xlabel='Actualizaciones de interfaz',ylabel='Pérdida holdout',title='Continuación desde750; selección cada250');axes[1].legend()
 fig.savefig('research/joint-extended-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {s} | {results[f'{s}-prior']['wins']}/{results[f'{s}-prior']['n']} | {results[f'{s}-control']['wins']}/{results[f'{s}-control']['n']} | {results[f'{s}-joint']['wins']}/{results[f'{s}-joint']['n']} | {steps[f'{s}-joint']} |" for s in SEEDS)
 a=comparisons['prior'];b=comparisons['control'];lo,hi=a['conditional_ci95_pp']
 doc=f'''# Extensión de interfaz750→3000

Principal frente a conjunto750: **{a['reference_mean_pct']:.2f}%→{a['joint_mean_pct']:.2f}%**, diferencia **{a['delta_pp']:+.2f}pp**, IC95% condicional [{lo:+.2f},{hi:+.2f}]. Secundario frente a control igualmente ampliado: **{b['delta_pp']:+.2f}pp**, IC95% {b['conditional_ci95_pp']}.

| Semilla | Conjunto750 | Control ampliado | Conjunto ampliado | Paso conjunto elegido |
|---|---|---|---|---|
{table}

![Resultados](joint-extended-results.png)

Tres pares reanudados exactamente desde últimos750,pesos/Adam/RNG preservados,2250updates adicionales por brazo. Mejores anteriores conservados en selección. Mismo dataset10000/holdout1000,lr/clipping,batch64/micro16,conectoma y ganancias fijos,sin DAgger adicional. Selección por menor pérdida cada250 sellada antes de500layouts finales nuevos7×7/7minas compartidos. {sum(reference)} aperturas ganadoras automáticas excluidas. No sustituir agente servido automáticamente.

Los intervalos usan10000 remuestreos pareados por tablero y promedian diferencias entre tres semillas; un cerebro,500tableros compartidos,no1500independientes. No demuestra ventaja anatómica ni compara directamente con DAgger~20% en otros benchmarks. Igual presupuesto de actualizaciones/control,distinto tiempo y parámetros entrenables.

Auditoría independiente reconstruye500layouts,solapamiento0,recalcula selección y coteja historial heredado/registros de restauración,pares de minibatches,duplicados por pesos,conteos,hashes y sello. Gradientes y restauración exacta verificados durante entrenamiento. Artefactos `runs/joint-extended-001` y `runs/joint-extended-SEED`; protocolo `research/PROTOCOLO_EXTENSION_INTERFAZ.md`.

```json
{json.dumps(summary,indent=2)}
```
'''
 Path('research/RESULTADO_EXTENSION_INTERFAZ.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
