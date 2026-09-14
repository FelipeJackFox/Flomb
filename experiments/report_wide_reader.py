import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_wide_reader import OUT,SEEDS,origin,digest
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 m=json.loads((OUT/'manifest.json').read_text());sel=json.loads((OUT/'selection.json').read_text());res=json.loads((OUT/'results.json').read_text());pair=json.loads((OUT/'pairing.json').read_text());games=json.loads((OUT/'games.json').read_text())
 assert all(digest(p)==sha for p,sha in m['hashes'].items());assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 keys={r['layout_hash'] for r in games};assert len(keys)==500
 for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent==OUT:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/*games.json'):
  if p.parent!=OUT:
   rows=json.loads(p.read_text())
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 assert not used&keys
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);wins={};reference=None
 for seed in SEEDS:
  assert pair[f'{seed}-local']==pair[f'{seed}-wide']
  for arm in ('baseline','local','wide'):
   key=f'{seed}-{arm}'
   if arm!='baseline':
    h=json.loads((OUT/f'history-{key}.json').read_text());assert [r['step'] for r in h]==list(range(0,1501,250));best=min(h,key=lambda r:r['validation_loss']);s=torch.load(OUT/f'best-{key}.pt',weights_only=False)
    assert s['step']==best['step'] and s['validation_loss']==best['validation_loss'] and s['dilation']==(1 if arm=='local' else 3)
    assert sel[key]['sha256']==digest(OUT/f'best-{key}.pt')
   rows=json.loads((OUT/f'{key}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected;auto=[r['automatic'] for r in rows]
   if reference is None:reference=auto
   assert auto==reference;played=[r for r in rows if not r['automatic']];wins[key]=np.array([int(r['won']) for r in played]);assert int(wins[key].sum())==res[key]['wins'] and len(played)==res[key]['n']
   for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==res[key][metric]
 comparisons={}
 for ref in ('local','baseline'):
  delta=np.stack([wins[f'{s}-wide']-wins[f'{s}-{ref}'] for s in SEEDS]).mean(0);rng=np.random.default_rng(20261024);boot=[rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)]
  comparisons[ref]=dict(reference_mean_pct=float(np.mean([wins[f'{s}-{ref}'].mean() for s in SEEDS])*100),wide_mean_pct=float(np.mean([wins[f'{s}-wide'].mean() for s in SEEDS])*100),delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(comparisons=comparisons,results=res,selection=sel,shared_layouts=500,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary);write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=500,prior_overlap=0,paired_draws_verified=True,selection_dilation_verified=True,hashes_and_seal_intact=True,counts_verified=True))
 fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');x=np.arange(3)
 for j,(arm,label,color) in enumerate([('baseline','Inicial','#89959e'),('local','Lector5×5','#a869af'),('wide','Lector9×9','#078d9d')]):
  vals=[100*wins[f'{s}-{arm}'].mean() for s in SEEDS];xx=x+(j-1)*.25;ax.bar(xx,vals,.25,label=label,color=color)
  for xp,v,s in zip(xx,vals,SEEDS):ax.text(xp,v+.5,str(res[f'{s}-{arm}']['wins']),ha='center',fontsize=9)
 ax.set_xticks(x,[str(s) for s in SEEDS]);ax.set_ylabel('Victorias autónomas (%)');ax.set_ylim(0,max(100*w.mean() for w in wins.values())+8);ax.legend();ax.set_title('Mismos parámetros · 500 tableros nuevos · 7×7 / 7 minas');fig.savefig('research/wide-reader-results.png',dpi=150);plt.close(fig)
 a=comparisons['local'];lo,hi=a['conditional_ci95_pp'];table='\n'.join(f"| {s} | {res[f'{s}-baseline']['wins']}/500 | {res[f'{s}-local']['wins']}/500 | {res[f'{s}-wide']['wins']}/500 |" for s in SEEDS)
 # Denominators may exclude automatic openings; use actual n in table.
 for s in SEEDS:
  for arm in ('baseline','local','wide'):
   r=res[f'{s}-{arm}'];table=table.replace(f"{r['wins']}/500",f"{r['wins']}/{r['n']}")
 doc=f'''# Lector con campo ampliado:5×5→9×9

Principal: local **{a['reference_mean_pct']:.2f}%**, amplio **{a['wide_mean_pct']:.2f}%**, diferencia **{a['delta_pp']:+.2f}pp**,IC95%condicional [{lo:+.2f},{hi:+.2f}].

| Semilla | Inicial | Local5×5 | Amplio9×9 |
|---|---|---|---|
{table}

![Resultados](wide-reader-results.png)

Trespares desdeDAggeradaptado,1500updatesextra,batch64(32originalesbalanceados+32DAggerheredado),mismosdraws/pesos/Adam/RNGiniciales. Encoder/cerebrofijos. Solo cambia dilation/padding segunda conv1→3,mismos12,769parámetros;test porgradiente valida campos5 y9. Cambia funcióninicial y contexto delosmomentos deAdam heredados,limitación importante. No eslector coninformación omnisciente:campo9noabarca todotablero desdecadaesquina,y activaciones ya mezclan informaciónrecurrencia.

Selección mínima pérdida0/cada250,sellada antes500layoutsfinalesnuevos7×7/7minas;{sum(reference)} aperturasautomáticas excluidas. Sin nuevastrayectorias ni etiquetas. Bootstrap10000 remuestreos por tablero,mediadediferenciasentretressemillas,un cerebro500compartidos,no1500independientes. No comparar absolutos conotrosbenchmarks ni concluir ventaja anatómica.

Auditoría independiente pasa:500layoutsreconstruidos,solapamiento0,batchespareados,selección/dilation/hashes/sello/conteos. Originales y agenteservido intactos. Protocolo `research/PROTOCOLO_LECTOR_AMPLIADO.md`,artefactos `runs/wide-reader-001`.

```json
{json.dumps(summary,indent=2)}
```
'''
 Path('research/RESULTADO_LECTOR_AMPLIADO.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
