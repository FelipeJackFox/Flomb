"""Audit pipeline adaptation contribution under matched DAgger collection layouts."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.compare_dagger_interfaces import OUT,SEEDS,unadapted,adapted,BASE,digest,tensor_hash
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 m=json.loads((OUT/'manifest.json').read_text());sel=json.loads((OUT/'selection.json').read_text());games=json.loads((OUT/'games.json').read_text());results=json.loads((OUT/'results.json').read_text())
 assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 assert all(digest(p)==sha for p,sha in m['hashes'].items()) and all(digest(r['path'])==r['sha256'] for r in sel.values())
 keys={r['layout_hash'] for r in games};assert len(keys)==500
 for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 cohort={OUT,*[unadapted(s) for s in SEEDS]};used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent in cohort:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/*games.json'):
  if p.parent in cohort:continue
  rows=json.loads(p.read_text())
  if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 assert not keys&used
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);wins={};reference=None;steps={}
 for seed in SEEDS:
  collection=json.loads((unadapted(seed)/'collection-games.json').read_text());assert collection==json.loads((adapted(seed)/'collection-games.json').read_text());assert len({r['layout_hash'] for r in collection})==900
  for r in collection:assert identity(Minesweeper(r['seed'],7,7))==r['layout_hash']
  assert not keys&{r['layout_hash'] for r in collection}
  for arm,folder,parentarm in [('unadapted',unadapted(seed),'control'),('adapted',adapted(seed),'joint')]:
   key=f'{seed}-{arm}';parent=torch.load(f'runs/joint-extended-{seed}/best-{parentarm}.pt',weights_only=False);saved=torch.load(folder/'best-dagger.pt',weights_only=False)
   hist=[dict(step=0,validation_loss=parent['validation_loss'])]+[r for r in json.loads((folder/'history.json').read_text()) if r['variant']=='dagger'];assert [r['step'] for r in hist]==list(range(0,2251,250))
   best=min(hist,key=lambda r:r['validation_loss']);assert saved['step']==best['step'] and abs(saved['validation_loss']-best['validation_loss'])<1e-5
   assert tensor_hash(saved['encoder'])==tensor_hash(parent['encoder']);steps[key]=saved['step']
   assert all(json.loads((folder/f'collection-{i}.json').read_text())['teacher_actions']==0 for i in (1,2,3))
   oldmanifest=json.loads((folder/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in oldmanifest['hashes'].items())
   rows=json.loads((OUT/f'{key}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected;auto=[r['automatic'] for r in rows]
   if reference is None:reference=auto
   assert reference==auto;played=[r for r in rows if not r['automatic']];wins[key]=np.array([int(r['won']) for r in played]);assert int(wins[key].sum())==results[key]['wins'] and len(played)==results[key]['n']
   for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==results[key][metric]
 comparisons={}
 for name,seeds in [('all',SEEDS),*[(str(s),(s,)) for s in SEEDS]]:
  d=np.stack([wins[f'{s}-adapted']-wins[f'{s}-unadapted'] for s in seeds]).mean(0);rng=np.random.default_rng(20261023);boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
  comparisons[name]=dict(unadapted_mean_pct=float(np.mean([wins[f'{s}-unadapted'].mean() for s in seeds])*100),adapted_mean_pct=float(np.mean([wins[f'{s}-adapted'].mean() for s in seeds])*100),delta_pp=float(d.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(comparisons=comparisons,results=results,steps=steps,shared_layouts=500,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary)
 write_json(OUT/'independent_verification.json',dict(final_layouts_reconstructed=500,prior_test_overlap=0,matched_collection_layouts_verified=True,selection_recomputed=True,encoder_frozen=True,teacher_actions=0,hashes_seal_counts_verified=True))
 fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');x=np.arange(3)
 for j,(arm,label,color) in enumerate([('unadapted','DAgger sin adaptación previa','#a56aa9'),('adapted','DAgger con adaptación previa','#078d9d')]):
  vals=[100*wins[f'{s}-{arm}'].mean() for s in SEEDS];xx=x+(j-.5)*.34;ax.bar(xx,vals,.34,label=label,color=color)
  for xp,v,s in zip(xx,vals,SEEDS):ax.text(xp,v+.5,str(results[f'{s}-{arm}']['wins']),ha='center')
 ax.set_xticks(x,[str(s) for s in SEEDS]);ax.set_ylabel('Victorias autónomas (%)');ax.set_ylim(0,max(100*w.mean() for w in wins.values())+9);ax.legend();ax.set_title('500 tableros nuevos compartidos · 7×7 / 7 minas');fig.savefig('research/dagger-interface-comparison-results.png',dpi=150);plt.close(fig)
 a=comparisons['all'];lo,hi=a['conditional_ci95_pp'];table='\n'.join(f"| {s} | {results[f'{s}-unadapted']['wins']}/{results[f'{s}-unadapted']['n']} | {results[f'{s}-adapted']['wins']}/{results[f'{s}-adapted']['n']} | {comparisons[str(s)]['delta_pp']:+.1f} |" for s in SEEDS)
 doc=f'''# Aporte de adaptación previa a DAgger

Sinadaptar **{a['unadapted_mean_pct']:.2f}%**, adaptando **{a['adapted_mean_pct']:.2f}%**. Diferencia adaptado−noadaptado **{a['delta_pp']:+.2f}pp**, IC95% condicional [{lo:+.2f},{hi:+.2f}].

| Semilla | DAgger sin adaptar | DAgger adaptado | Diferencia(pp) |
|---|---|---|---|
{table}

![Comparación](dagger-interface-comparison-results.png)

Comparación de pipelines completos, no ablación aislada delencoder. Ambas etapasprevias tuvieron3000updates nominales para selección de checkpoints; adaptarentrada también cambia lector/Adam/RNG y cuesta más tiempo. DAgger despuéscongelóencoderycerebro y entrenó lector2250updates con iguales900layoutsdecolección porpareja,misma mezcla50%original50%experiencia. Trayectorias y datosresultantes pueden diferir. Losadaptados existentes se preservaron;losnoadaptados se entrenaronahora desdecontrolprevio. Misma pérdida/regladeholdout,sello antesdefinal500layouts nuevosdisjuntos. {sum(reference)} aperturasganadoras automáticas excluidas.

Bootstrap10000 remuestreos pareadosportablero,promedioentresemillas,condicional a estosmodelos;un cerebro500tableroscompartidos,no1500independientes. No demuestra ventaja anatómica ni generalidad. No comparar absolutosconotrosbenchmarks. Maestroacciones0.

Auditoría final500layoutsreconstruidos/solapamiento0,colección900 porpareja coincidey reconstruida,selección/encoderfijo/teacheracciones0/hashes/sello/conteosverificados. Originales preservados,ningún agenteservido cambiado. Protocolo `research/PROTOCOLO_COMPARACION_INTERFACES_DAGGER.md`;artefactos `runs/dagger-interface-comparison-001`.

```json
{json.dumps(summary,indent=2)}
```
'''
 Path('research/RESULTADO_COMPARACION_INTERFACES_DAGGER.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
