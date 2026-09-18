import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_risk_auxiliary import OUT,PARENT,BASE,SEED,origin,digest
from experiments.risk_auxiliary import mine_labels
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 m=json.loads((OUT/'manifest.json').read_text());results=json.loads((OUT/'results.json').read_text());sel=json.loads((OUT/'selection.json').read_text());games=json.loads((OUT/'games.json').read_text());pair=json.loads((OUT/'pairing.json').read_text())
 assert pair['control']==pair['auxiliary'] and all(digest(p)==sha for p,sha in m['hashes'].items());assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
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
 assert not keys&used
 data=pickle.loads((BASE/'dataset.pkl').read_bytes());extra=pickle.loads((origin(SEED)/'dataset.pkl').read_bytes())['train'];labels=torch.load(OUT/'mine-labels.pt',weights_only=False)
 for name,rows in [('original',data['train']),('experience',extra),('holdout',data['holdout'])]:torch.testing.assert_close(labels[name],mine_labels(rows),rtol=0,atol=0)
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);wins={};reference=None
 for arm in ('baseline','control','auxiliary'):
  if arm!='baseline':
   h=json.loads((OUT/f'history-{arm}.json').read_text());assert [r['step'] for r in h]==list(range(0,1501,250));b=min(h,key=lambda r:r['validation_loss']);saved=torch.load(OUT/f'best-{arm}.pt',weights_only=False)
   assert saved['step']==b['step'] and saved['validation_loss']==b['validation_loss'] and digest(OUT/f'best-{arm}.pt')==sel[arm]['sha256']
  rows=json.loads((OUT/f'{arm}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected;auto=[r['automatic'] for r in rows]
  if reference is None:reference=auto
  assert reference==auto;played=[r for r in rows if not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played]);assert int(wins[arm].sum())==results[arm]['wins'] and len(played)==results[arm]['n']
  for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==results[arm][metric]
 comp={}
 for ref in ('control','baseline'):
  d=wins['auxiliary']-wins[ref];rng=np.random.default_rng(20261025);boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
  comp[ref]=dict(delta_pp=float(d.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(results=results,comparisons=comp,selection=sel,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary);write_json(OUT/'independent_verification.json',dict(final_layouts_reconstructed=500,prior_overlap=0,public_certificates_recomputed=len(data['train'])+len(extra)+len(data['holdout']),paired_draws=True,selection_hashes_seal_counts_verified=True))
 fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');names={'baseline':'Inicial','control':'Solo política','auxiliary':'Política + seguridad'}
 for i,arm in enumerate(names):
  v=100*wins[arm].mean();axes[0].bar(i,v,color=['#8b979f','#a769ad','#078d9d'][i]);axes[0].text(i,v+.5,f"{results[arm]['wins']}/{results[arm]['n']}",ha='center')
 axes[0].set_xticks(range(3),list(names.values()));axes[0].set_ylabel('Victorias autónomas (%)');axes[0].set_ylim(0,max(100*w.mean() for w in wins.values())+8);axes[0].set_title('Piloto:500 tableros nuevos7×7/7minas')
 for arm in ('control','auxiliary'):
  h=json.loads((OUT/f'history-{arm}.json').read_text());axes[1].plot([r['step'] for r in h],[r['validation_loss'] for r in h],'-o',label=names[arm])
 axes[1].set(xlabel='Updates adicionales',ylabel='Pérdida de política',title='Mismo criterio de selección');axes[1].legend();fig.savefig('research/risk-auxiliary-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {names[a]} | {results[a]['wins']}/{results[a]['n']} ({100*wins[a].mean():.1f}%) | {results[a]['known_mine_choices']} |" for a in names);c=comp['control'];lo,hi=c['conditional_ci95_pp']
 doc=f'''# Auxiliar de seguridad: piloto

Auxiliar−control **{c['delta_pp']:+.2f}pp**,IC95%condicional [{lo:+.2f},{hi:+.2f}]. Una semilla/un cerebro;no consistencia todavía.

| Variante | Victorias | Clics en minas deducibles |
|---|---|---|
{table}

![Resultados](risk-auxiliary-results.png)

Mismos pesos/Adam/RNG/batches/1500updates,mezcla32original+32DAggerheredado;encoder/cerebrocongelados. CEpolítica vsCE+0.2BCEcertificadossafe/mine,ramaaux33parámetros,ceroinicial. BCEbalancea clasespresentes;casillasinciertasnoetiquetadas. Maestro nofiltra acciones;inferencias usan sololector5×5original12,769parámetros. Tests confirman gradientesinciertas0,signoscorrectos yforwardpolíticainalteradoaladjuntarrama. CEya penalizaminas indirectamente;esto agregaobjetivo distinto,no muestreo prioritario.

Selecciónpor mismaCEpolítica enholdout0/cada250,sello antesde500testnuevos;{sum(reference)} aperturasautomáticasexcluidas. Métricasauxholdout son diagnóstico,no selección,controlauxceronoentrenado no es comparación de capacidad. Intervalos10000remuestreos pareados portablero. No comparar absolutosconotrascorridas. Auditó500layouts nuevos/solapamiento0,certificados públicos recomputados,todosdraws/selección/hashes/sello/conteos. Originales yservido intactos.

Protocolo `research/PROTOCOLO_AUXILIAR_RIESGO.md`,artefactos `runs/risk-auxiliary-001`.

```json
{json.dumps(summary,indent=2)}
```
'''
 Path('research/RESULTADO_AUXILIAR_RIESGO.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
