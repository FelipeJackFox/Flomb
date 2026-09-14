import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.select_by_games import OUT,SEEDS
from experiments.capacity_probe import write_json

def main():
 assert (OUT/'completed.json').exists()
 manifest=json.loads((OUT/'manifest.json').read_text());sel=json.loads((OUT/'selection.json').read_text());test=json.loads((OUT/'test-results.json').read_text());val=json.loads((OUT/'validation-results.json').read_text());metrics={};wins={};ref=None
 for arm,choice in sel.items():
  metrics[arm]={};wins[arm]={}
  for mode,key in choice.items():
   rr=json.loads((OUT/test[key]['rows_file']).read_text());ids=[(r['seed'],r['layout_hash'],r['automatic']) for r in rr]
   if ref is None:ref=ids
   assert ids==ref;played=[r for r in rr if not r['automatic']];w=np.array([int(r['won']) for r in played]);wins[arm][mode]=w
   metrics[arm][mode]=dict(candidate=key,step=manifest['candidates'][key]['step'],wins=int(w.sum()),n=len(w),validation_wins=val[key]['wins'],validation_n=val[key]['n'],known_mine_choices=sum(r['known_mine_choices'] for r in played),death_with_safe_available=sum(r['death_with_safe_available'] for r in played),safe_choices=sum(r['safe_choices'] for r in played),safe_opportunities=sum(r['safe_opportunities'] for r in played))
 rng=np.random.default_rng(20261017);comparisons={}
 for group in ('control','priority','all'):
  arms=[a for a in wins if group=='all' or a.endswith(group)];delta=np.stack([wins[a]['games']-wins[a]['loss'] for a in arms]).mean(0);boot=[rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)]
  comparisons[group]=dict(mean_delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),loss_mean_pct=float(np.mean([wins[a]['loss'].mean() for a in arms])*100),game_mean_pct=float(np.mean([wins[a]['games'].mean() for a in arms])*100))
 summary=dict(comparisons=comparisons,metrics=metrics,automatic_excluded=sum(r[2] for r in ref));write_json(OUT/'summary.json',summary)
 fig,ax=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained');x=np.arange(3)
 for axis,group,label in zip(ax,('control','priority'),('Muestreo uniforme','Muestreo prioritario')):
  for j,(mode,title,color) in enumerate([('loss','Selección por pérdida','#89959e'),('games','Selección por partidas','#078493')]):
   xx=x+(j-.5)*.32;vv=[100*wins[f'{s}-{group}'][mode].mean() for s in SEEDS];axis.bar(xx,vv,.32,label=title,color=color)
   for xp,v,s in zip(xx,vv,SEEDS):axis.text(xp,v+.25,str(metrics[f'{s}-{group}'][mode]['wins']),ha='center',fontsize=10)
  axis.set_ylim(0,max(100*wins[a][m].mean() for a in wins for m in ('loss','games'))*1.35);axis.set_xticks(x,[str(s) for s in SEEDS]);axis.set_title(label);axis.set_ylabel('Victorias finales (%)');axis.legend(fontsize=9)
 fig.savefig('research/game-selection-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {a} | {r['loss']['candidate']} ({r['loss']['step']}) | {r['games']['candidate']} ({r['games']['step']}) | {r['loss']['wins']}/{r['loss']['n']} | {r['games']['wins']}/{r['games']['n']} |" for a,r in metrics.items())
 doc=f'''# Selección de checkpoints por partidas de validación

La selección por partidas no mostró una mejora consistente en esta prueba: media final 18.90% frente a 19.20% por pérdida, diferencia −0.30 puntos porcentuales e IC95% condicional [−1.50,+0.87]. Mejoró dos comparaciones, empeoró tres y empató una. No justifica sustituir el criterio anterior; tampoco demuestra que la pérdida sea óptima ni descarta otros diseños de validación.

Se comparan dos reglas usando checkpoints existentes: mínimo de pérdida original frente a máximo de victorias autónomas en un nuevo conjunto de validación. No se entrenó ningún peso. El conjunto final es distinto del de selección y de todos los anteriores.

| Brazo | Elegido por pérdida (update) | Elegido por partidas (update) | Final: pérdida | Final: partidas |
|---|---|---|---|---|\n{table}

![Comparación final](game-selection-results.png)

## Diferencias emparejadas finales

```json\n{json.dumps(comparisons,indent=2)}\n```

Intervalos de 10,000 remuestreos por tablero, promediando diferencias dentro de cada tablero entre las tres semillas del brazo (o los seis casos para `all`). Condicionados a estos lectores, no a todos los entrenamientos posibles. Los dos brazos comparten modelos iniciales y un cerebro; no son seis cerebros independientes. Hay {len(ref)} layouts finales compartidos, {sum(r[2] for r in ref)} aperturas ganadoras automáticas excluidas, y {len(ref)-sum(r[2] for r in ref)} partidas con decisiones.

## Qué se seleccionó y qué significa

Cada brazo tiene tres candidatos disponibles: modelo anterior, mejor por pérdida guardado y último guardado. No se recuperaron checkpoints intermedios inexistentes ni se reentrenó. Si dos candidatos tienen pesos idénticos, comparten evaluación exacta. La regla por partidas maximiza victorias en 500 layouts de validación; desempata por menos actualizaciones adicionales y luego orden fijo. Se excluyen aperturas ganadoras automáticas idénticamente entre modelos.

Las elecciones de los seis brazos se guardaron y sellaron antes de evaluar otros 500 layouts finales nuevos. No se consultaron esos resultados finales para escoger ni se evaluaron allí candidatos descartados para retocar la elección. La comparación por pérdida conserva el checkpoint de la fase anterior; el nuevo criterio se compara dentro de ese mismo brazo, no eligiendo además un algoritmo ganador.

Un mejor porcentaje en validación puede deberse parcialmente a selección sobre ruido. La mejora que importa es la del conjunto final independiente. Esta prueba mide una selección entre tres checkpoints disponibles, no prueba que el entrenamiento ni la representación cerebral se hayan arreglado.

## Métricas por elección

```json\n{json.dumps(metrics,indent=2)}\n```

`validation_wins` corresponde al conjunto de selección; `wins` al conjunto final. No mezclar sus porcentajes. Las métricas de minas conocidas y alternativas seguras describen trayectorias distintas, con denominadores variables.

Protocolo research/PROTOCOLO_SELECCION_POR_PARTIDAS.md. Fuentes, hashes, candidatos, resultados individuales y sello en runs/game-selection-001. La prueba de selección verificó victorias y desempates. Auditoría independiente reconstruyó los 1,000 layouts, comprobó ausencia de solapamientos, recalculó las seis elecciones, cotejó conteos por partida y pesos deduplicados, y verificó el sello y los hashes originales. Se evaluaron 14 conjuntos de pesos distintos en validación y 11 en prueba final. No se modificaron checkpoints ni se sustituyó el agente servido.
'''
 Path('research/RESULTADO_SELECCION_POR_PARTIDAS.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
