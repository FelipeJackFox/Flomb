"""Independent pilot audit and paired outcome report."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_joint_interface import OUT,BASE,PARENT,digest,tensor_hash
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
 assert (OUT/'completed.json').exists()
 manifest=json.loads((OUT/'manifest.json').read_text());games=json.loads((OUT/'games.json').read_text());selection=json.loads((OUT/'selection.json').read_text());results=json.loads((OUT/'results.json').read_text())
 keys={r['layout_hash'] for r in games};assert len(keys)==500
 for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
 used=set()
 for p in Path('runs').glob('*/dataset.pkl'):
  if p.parent==OUT:continue
  for rows in pickle.loads(p.read_bytes()).values():
   if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 for p in Path('runs').glob('*/*games.json'):
  if p.parent==OUT:continue
  rows=json.loads(p.read_text())
  if isinstance(rows,list):used.update(r['layout_hash'] for r in rows if isinstance(r,dict) and 'layout_hash' in r)
 assert not keys&used
 assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
 assert digest(OUT/'selection.json')==json.loads((OUT/'selection-seal.json').read_text())['sha256']
 pairing=json.loads((OUT/'pairing.json').read_text());assert pairing['control']==pairing['joint']
 source=torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model'];encoder={k.removeprefix('encoder.'):v for k,v in source.items() if k.startswith('encoder.')}
 changes={}
 for arm in ('control','joint'):
  h=json.loads((OUT/f'history-{arm}.json').read_text());assert [r['step'] for r in h]==[0,250,500,750]
  best=min(h,key=lambda r:r['validation_loss']);assert selection[arm]==dict(step=best['step'],validation_loss=best['validation_loss'])
  for kind in ('best','latest'):
   saved=torch.load(OUT/f'{kind}-{arm}.pt',weights_only=False)
   if kind=='best':assert saved['step']==best['step'] and saved['validation_loss']==best['validation_loss']
   changes[f'{kind}-{arm}']=sum(float((v-encoder[k]).square().sum()) for k,v in saved['encoder'].items())**.5
   if arm=='control':assert tensor_hash(saved['encoder'])==tensor_hash(encoder)
 assert changes['latest-joint']>0
 expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games);reference=None;wins={}
 for arm in ('baseline','control','joint'):
  rows=json.loads((OUT/f'{arm}-games.json').read_text());assert [(r['size'],r['seed'],r['layout_hash']) for r in rows]==expected
  auto=[r['automatic'] for r in rows]
  if reference is None:reference=auto
  assert reference==auto
  played=[r for r in rows if not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
  assert len(played)==results[arm]['n'] and int(wins[arm].sum())==results[arm]['wins']
  for metric in ('known_mine_choices','safe_choices','safe_opportunities'):assert sum(r[metric] for r in played)==results[arm][metric]
 rng=np.random.default_rng(20261019);comparisons={}
 for ref in ('baseline','control'):
  rng=np.random.default_rng(20261019)  # Identical references receive identical resamples.
  delta=wins['joint']-wins[ref];boot=[rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)]
  comparisons[ref]=dict(joint_minus_reference_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 summary=dict(comparisons=comparisons,results=results,encoder_change_l2=changes,n_shared_games=500,automatic_excluded=sum(reference))
 write_json(OUT/'summary.json',summary)
 write_json(OUT/'independent_verification.json',dict(layouts_reconstructed=500,prior_overlap=0,selection_rule_recomputed=True,paired_draws_equal=True,encoder_changes_verified=True,result_counts_verified=True,seal_intact=True,original_hashes_unchanged=True))
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained');labels={'baseline':'Inicial','control':'Lector solo','joint':'Entrada + lector'};colors={'baseline':'#8b969f','control':'#a968aa','joint':'#078d9d'}
 for i,arm in enumerate(labels):
  pct=100*wins[arm].mean();axes[0].bar(i,pct,color=colors[arm]);axes[0].text(i,pct+.7,f"{results[arm]['wins']}/{results[arm]['n']}",ha='center')
 axes[0].set_xticks(range(3),list(labels.values()));axes[0].set_ylabel('Victorias autónomas (%)');axes[0].set_ylim(0,max(100*v.mean() for v in wins.values())+8);axes[0].set_title('Piloto: una semilla · 7×7 / 7 minas')
 for arm in ('control','joint'):
  h=json.loads((OUT/f'history-{arm}.json').read_text());axes[1].plot([r['step'] for r in h],[r['validation_loss'] for r in h],'-o',label=labels[arm],color=colors[arm])
 axes[1].set(xlabel='Actualizaciones adicionales',ylabel='Pérdida de validación',title='Selección previa a partidas finales');axes[1].legend()
 fig.savefig('research/joint-interface-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {labels[a]} | {results[a]['wins']}/{results[a]['n']} ({100*wins[a].mean():.1f}%) | {0 if a=='baseline' else selection[a]['step']} | {results[a]['known_mine_choices']} |" for a in labels)
 lo,hi=comparisons['control']['conditional_ci95_pp'];delta=comparisons['control']['joint_minus_reference_pp']
 doc=f'''# Entrenamiento conjunto de entrada y lector: piloto

Diferencia conjunto menos control: **{delta:+.2f} puntos porcentuales**, IC95% condicional [{lo:+.2f},{hi:+.2f}]. Una semilla; no establece consistencia entre entrenamientos.

| Variante | Victorias autónomas | Updates adicionales elegidos | Clics en minas deducibles |
|---|---|---|---|
{table}

![Resultado del piloto](joint-interface-results.png)

Ambos brazos continúan el mismo lector cerebral elegido en3500updates, con Adam/RNG heredados y los mismos750 minibatches64. Conjunto añade encoder entrenable lr0.0001, lector lr0.001 en ambos. Acumula4microbatches16; clipping5 separado para lector y encoder. Se mantuvieron conexiones, signos, ganancias neuronales, dinámica, mapeo y pooling fijos. Solo control usa caché de actividad para entrenar. Sin nueva experiencia DAgger, mismos10000 ejemplos/1000holdout históricos. Selección mínima pérdida0/250/500/750, incluye inicial; sello previo a prueba final.

500 layouts nuevos compartidos7×7/7minas, {sum(reference)} aperturas ganadoras automáticas excluidas. Intervalos de10000 remuestreos pareados por tablero, condicionados a una semilla y un cerebro. El control tiene menos parámetros entrenables y menor tiempo de cómputo: igualamos ejemplos/actualizaciones, no tiempo. Maestro no filtra decisiones autónomas. Las métricas por trayectoria tienen denominadores distintos.

Una mejora sería evidencia piloto a favor de adaptar la interfaz actual; no demostraría ventaja anatómica ni superioridad general. Una ausencia de mejora tampoco descartaría otros presupuestos, tasas o lectores. No comparar directamente con las corridas DAgger de~20% ni con otros tableros finales. Ningún modelo servido fue reemplazado.

## Auditoría y optimización

Ruta autograd de pesos fijos: reutiliza CSR y transpuesta, sin gradientes de aristas. Test de encoder frente a gradiente denso pasa. Grafo real: forward equivalente a ruta previa, gradientes de acumulación equivalentes a minibatch completo, gradiente encoder no nulo, validación inicial online/caché coincide, pesos restaurados producen igual forward con memo. Resto del cerebro intacto; encoder control intacto y conjunto modificado. Auditoría independiente reconstruye500 layouts y verifica solapamiento0, selección, conteos, cambios encoder, sello y hashes originales. Pesos mejores/últimos con Adam/RNG conservados.

```json
{json.dumps(summary,indent=2)}
```

Protocolo `research/PROTOCOLO_INTERFAZ_CONJUNTA.md`, artefactos `runs/joint-interface-001`.
'''
 Path('research/RESULTADO_INTERFAZ_CONJUNTA.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
