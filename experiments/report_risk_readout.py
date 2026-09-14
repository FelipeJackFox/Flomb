import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.evaluate_risk_readout import OUT,LAST,PARENT,digest,tensor_hash
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 m=json.loads((OUT/'manifest.json').read_text());results=json.loads((OUT/'results.json').read_text());games=json.loads((OUT/'games.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
 last=torch.load(LAST,weights_only=False);parent=torch.load(PARENT,weights_only=False);assert last['step']==1500 and tensor_hash(last['encoder'])==tensor_hash(parent['encoder'])
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
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);wins={};reference=None
 for arm in ('baseline','last_policy','last_risk'):
  rows=json.loads((OUT/f'{arm}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected;auto=[r['automatic'] for r in rows]
  if reference is None:reference=auto
  assert auto==reference;played=[r for r in rows if not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played]);assert int(wins[arm].sum())==results[arm]['wins'] and len(played)==results[arm]['n']
  for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==results[arm][metric]
 comparisons={}
 for ref in ('last_policy','baseline'):
  d=wins['last_risk']-wins[ref];rng=np.random.default_rng(20261026);boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
  comparisons[ref]=dict(delta_pp=float(d.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(results=results,comparisons=comparisons,shared_layouts=500,automatic_excluded=sum(reference));write_json(OUT/'summary.json',summary);write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=500,prior_overlap=0,last_checkpoint_fixed=True,encoder_identical=True,hashes_intact=True,result_counts_verified=True))
 names={'baseline':'Inicial preservado','last_policy':'Último: política','last_risk':'Último: riesgo'};fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
 for i,arm in enumerate(names):
  v=100*wins[arm].mean();ax.bar(i,v,color=['#8e979e','#a667ab','#078d9d'][i]);ax.text(i,v+.5,f"{results[arm]['wins']}/{results[arm]['n']}",ha='center')
 ax.set_xticks(range(3),list(names.values()));ax.set_ylabel('Victorias autónomas (%)');ax.set_ylim(0,max(100*w.mean() for w in wins.values())+8);ax.set_title('Dos salidas del mismo checkpoint · 500 tableros nuevos');fig.savefig('research/risk-readout-results.png',dpi=150);plt.close(fig)
 c=comparisons['last_policy'];lo,hi=c['conditional_ci95_pp'];table='\n'.join(f"| {names[a]} | {results[a]['wins']}/{results[a]['n']} ({100*wins[a].mean():.1f}%) | {results[a]['known_mine_choices']} |" for a in names)
 doc=f'''# Decidir por riesgo aprendido

Comparación principal, riesgo menos política del mismo último checkpoint: **{c['delta_pp']:+.2f} puntos porcentuales**, IC95% condicional [{lo:+.2f}, {hi:+.2f}].

| Salida | Victorias | Clics en minas deducibles |
|---|---|---|
{table}

![Resultados](risk-readout-results.png)

No hubo entrenamiento adicional. Se fijó el último auxiliar1500 antes de la prueba, comparando su salida de política con escoger el menor logit de mina aprendido. Ambos usan idéntico encoder y representación compartida; cambia únicamente la salida que decide. El baseline inicial es una comparación secundaria. No se seleccionaron checkpoints según estas partidas.

No hay filtro ni decisiones del maestro; solo máscara de acciones legales. La salida de riesgo, entrenada con clases balanceadas y sin objetivos para inciertas, no es probabilidad calibrada de mina. El desempeño en casillas inciertas es una limitación central. Test confirma que cambiar la capa final de política no altera la salida de riesgo.

500 layouts nuevos7×7/7minas compartidos, {sum(reference)} aperturas ganadoras automáticas excluidas. Intervalos de10000 remuestreos pareados por tablero; una semilla/un cerebro, no consistencia entre entrenamientos. Auditoría reconstruye500 layouts y verifica solapamiento0, checkpoint fijo, encoder, hashes y conteos. Originales y agente servido intactos.

Protocolo `research/PROTOCOLO_LECTURA_RIESGO.md`; artefactos `runs/risk-readout-001`.

```json
{json.dumps(summary,indent=2)}
```
'''
 Path('research/RESULTADO_LECTURA_RIESGO.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
