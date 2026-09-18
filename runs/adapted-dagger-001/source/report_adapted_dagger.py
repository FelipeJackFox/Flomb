"""Audit and report the frozen adapted-interface DAgger pilot."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.adapted_dagger import OUT,BASE,digest
from experiments.train_joint_interface import tensor_hash
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 m=json.loads((OUT/'manifest.json').read_text());results=json.loads((OUT/'results.json').read_text());sel=json.loads((OUT/'selection.json').read_text());history=json.loads((OUT/'history.json').read_text())
 assert all(digest(p)==sha for p,sha in m['hashes'].items());assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 games=json.loads((OUT/'games.json').read_text());collection=json.loads((OUT/'collection-games.json').read_text());keys={r['layout_hash'] for r in games+collection};assert len(games)==500 and len(collection)==900 and len(keys)==1400
 for r in games+collection:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent==OUT:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/*games.json'):
  if p.parent!=OUT:
   rows=json.loads(p.read_text())
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 assert not keys&used
 parent=torch.load('runs/joint-extended-20261002/best-joint.pt',weights_only=False)
 for arm in ('control','dagger'):
  candidates=[dict(step=0,validation_loss=parent['validation_loss']),*[r for r in history if r['variant']==arm]];best=min(candidates,key=lambda r:r['validation_loss']);saved=torch.load(OUT/f'best-{arm}.pt',weights_only=False)
  assert saved['step']==best['step'] and abs(saved['validation_loss']-best['validation_loss'])<1e-5
  assert sel[arm]['sha256']==digest(OUT/f'best-{arm}.pt') and sel[arm]['step']==saved['step']
  assert tensor_hash(saved['encoder'])==tensor_hash(parent['encoder'])
 data=pickle.loads((OUT/'dataset.pkl').read_bytes())['train'];trainkeys={r['layout_hash'] for r in collection}
 assert all(r['layout_hash'] in trainkeys for r in data)
 assert all(json.loads((OUT/f'collection-{i}.json').read_text())['teacher_actions']==0 for i in (1,2,3))
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);reference=None;wins={};metrics={}
 for arm in ('baseline','control','dagger'):
  rows=json.loads((OUT/f'{arm}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected;auto=[r['automatic'] for r in rows]
  if reference is None:reference=auto
  assert auto==reference;played=[r for r in rows if not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played]);assert int(wins[arm].sum())==results[arm]['wins'] and len(played)==results[arm]['n']
  metrics[arm]={k:sum(r[k] for r in played) for k in ('known_mine_choices','safe_choices','safe_opportunities')}
 comparisons={}
 for arm in ('control','baseline'):
  rng=np.random.default_rng(20261022);d=wins['dagger']-wins[arm];boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
  comparisons[arm]=dict(delta_pp=float(d.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(results=results,comparisons=comparisons,metrics=metrics,new_positions=len(data),shared_layouts=500,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary)
 write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=1400,prior_overlap=0,selection_recomputed=True,encoder_unchanged=True,teacher_actions=0,counts_verified=True,hashes_and_seal_intact=True))
 labels={'baseline':'Inicial adaptado','control':'Solo datos anteriores','dagger':'DAgger + adaptado'};fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
 for i,arm in enumerate(labels):
  v=100*wins[arm].mean();ax.bar(i,v,color=['#8e999e','#a86db2','#078d9d'][i]);ax.text(i,v+.5,f"{results[arm]['wins']}/{results[arm]['n']}",ha='center')
 ax.set_xticks(range(3),list(labels.values()));ax.set_ylabel('Victorias autónomas (%)');ax.set_ylim(0,max(100*w.mean() for w in wins.values())+8);ax.set_title('Piloto DAgger con interfaz adaptada · 500 tableros nuevos');fig.savefig('research/adapted-dagger-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {labels[a]} | {results[a]['wins']}/{results[a]['n']} ({100*wins[a].mean():.1f}%) | {results[a]['selected_step']} |" for a in labels)
 c=comparisons['control'];lo,hi=c['conditional_ci95_pp']
 doc=f'''# DAgger con interfaz adaptada: piloto

DAgger−control: **{c['delta_pp']:+.2f} puntos**, IC95% condicional [{lo:+.2f},{hi:+.2f}]. Una semilla/un cerebro; no establece consistencia.

| Variante | Victorias | Updates elegidos |
|---|---|---|
{table}

![Comparación](adapted-dagger-results.png)

Desde interfaz adaptada2750 de20261002,encoder/cerebro congelados y lector entrenable. Adam del lector/RNG conservados;test exacto de unpaso al retirargrupo encoder pasa. Tres rondas300layouts de colección nuevos+750updates,control64originales frente a32originales+32agregadoDAgger uniforme. {len(data)} posiciones nuevas con acciones seguras certificadas;el agente toma todas las acciones,maestro solo etiqueta. Cache recomputada del encoderadaptado; pérdida de validación inicial coincide con elcheckpoint padre.

Selección por holdout original cada250,incluyeinicial,sellada antes de500layouts finales nuevos7×7/7minas;{sum(reference)} aperturasautomáticas excluidas. Bootstrap10000 remuestreos pareados por tablero,condicional a estos modelos. No comparar directamente con modelos DAgger de otros benchmarks ni interpretar éxito asistido. Sin filtro del maestro al actuar.

Auditoría reconstruye1400layouts únicos,solapamiento previo0,selecciónrecalculada,encoderintacto,teacheracciones0,conteos/hashes/sello intactos. Nuevos datos provienen de colección y nunca del conjunto final. Originales preservados,ningún agenteservido sustituido. Protocolo `research/PROTOCOLO_DAGGER_INTERFAZ_ADAPTADA.md`,artefactos `runs/adapted-dagger-001`.

```json
{json.dumps(summary,indent=2)}
```
'''
 Path('research/RESULTADO_DAGGER_INTERFAZ_ADAPTADA.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
