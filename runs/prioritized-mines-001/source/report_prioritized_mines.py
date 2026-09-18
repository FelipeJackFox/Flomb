import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.prioritize_mine_errors import OUT,target,SEEDS
from experiments.capacity_probe import write_json

def main():
 assert (OUT/'completed.json').exists()
 names=('baseline','control','priority');labels=('Anterior','Uniforme','Prioritario');metrics={};wins={};ref=None
 for s in SEEDS:
  metrics[s]={};wins[s]={}
  for n in names:
   rr=json.loads((target(s)/f'{n}-games.json').read_text());keys=[(r['seed'],r['layout_hash'],r['automatic']) for r in rr]
   if ref is None:ref=keys
   assert keys==ref and len(keys)==len(set(keys))==500
   rr=[r for r in rr if not r['automatic']];w=np.array([int(r['won']) for r in rr]);wins[s][n]=w;metrics[s][n]=dict(wins=int(w.sum()),n=len(w),known_mine_choices=sum(r['known_mine_choices'] for r in rr),death_with_safe_available=sum(r['death_with_safe_available'] for r in rr),safe_choices=sum(r['safe_choices'] for r in rr),safe_opportunities=sum(r['safe_opportunities'] for r in rr))
 rng=np.random.default_rng(20261015);ci={}
 for other in ('baseline','control'):
  delta=np.stack([wins[s]['priority']-wins[s][other] for s in SEEDS]).mean(0);boot=[rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)]
  ci[other]=dict(delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
 fit={s:json.loads((target(s)/'fit-metrics.json').read_text()) for s in SEEDS};summary=dict(metrics=metrics,comparisons=ci,fit=fit,terminal_fit=json.loads((OUT/'terminal-fit.json').read_text()),automatic_excluded=sum(k[2] for k in ref),means_pct={n:float(np.mean([wins[s][n].mean() for s in SEEDS])*100) for n in names});write_json(OUT/'summary.json',summary)
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(3)
 for i,(n,label,color) in enumerate(zip(names,labels,('#87929d','#d6a04b','#078493'))):
  xx=x+(i-1)*.25;vv=[100*wins[s][n].mean() for s in SEEDS];ax[0].bar(xx,vv,.25,label=label,color=color)
  for xp,v,s in zip(xx,vv,SEEDS):ax[0].text(xp,v+.3,str(metrics[s][n]['wins']),ha='center',fontsize=9)
  ax[1].bar(xx,[metrics[s][n]['known_mine_choices'] for s in SEEDS],.25,label=label,color=color)
 for a in ax:a.set_xticks(x,[str(s) for s in SEEDS]);a.legend(fontsize=9)
 ax[0].set_ylim(0,max(wins[s][n].mean()*100 for s in SEEDS for n in names)*1.4);ax[0].set_ylabel('Victorias (%)');ax[0].set_title('499 partidas con decisiones · 7×7');ax[1].set_ylabel('Clics en minas deducibles');ax[1].set_title('Errores durante partidas autónomas');ax[1].set_ylim(0,max(metrics[s][n]['known_mine_choices'] for s in SEEDS for n in names)*1.3)
 fig.savefig('research/prioritized-mines-results.png',dpi=150);plt.close(fig)
 table='\n'.join('| '+str(s)+' | '+' | '.join(f"{metrics[s][n]['wins']}/{metrics[s][n]['n']} ({100*wins[s][n].mean():.1f}%)" for n in names)+' |' for s in SEEDS)
 doc=f'''# Priorizar errores de minas deducibles

Esta configuración no mostró mejora reproducible. Media de victorias ajustada por aperturas automáticas: anterior 19.37%, uniforme 19.44%, prioritario 18.84%. Diferencia priorizado−uniforme −0.60 puntos, IC95% condicional [−2.34,+1.14]. No demuestra que toda priorización sea inútil ni una caída estadísticamente estable; no justifica adoptar esta variante.

Se evaluaron 500 layouts nuevos; uno terminó ganado en la apertura central sin decisión del modelo. Se excluye idénticamente en los nueve modelos: los resultados siguientes usan 499 partidas con decisiones. Las cifras de progreso de consola incluían esa victoria automática.

| Semilla | Anterior | Uniforme | Prioritario |
|---|---|---|---|\n{table}

![Resultados](prioritized-mines-results.png)

## Comparación emparejada de victorias

```json\n{json.dumps(ci,indent=2)}\n```

Intervalos bootstrap de 10,000 remuestreos por tablero, promediando primero las diferencias entre lectores en cada tablero. Condicionados a estas tres semillas sobre un único cerebro. Son 499 partidas con decisiones de 500 layouts compartidos, no 1,500 ensayos independientes.

## Ajuste sobre los datos disponibles

```json\n{json.dumps(fit,indent=2)}\n```

`initial_known_mines` y `selected_known_mines` miden elecciones de minas certificadas sobre el buffer DAgger, con alternativa segura disponible; no son partidas. `selected_step=0` conserva el modelo anterior. La lista prioritaria se recalcula cada 250 actualizaciones, por lo que puede incluir errores corregidos dentro de ese bloque.

## Ajuste final frente a selección

Los últimos checkpoints prioritarios redujeron las elecciones de mina sobre el buffer frente a los últimos uniformes: 579 vs619, 387 vs490 y381 vs481. Sin embargo, no siempre aumentaron las decisiones seguras totales y no fueron los seleccionados. La selección por pérdida eligió prioridad en pasos500/1000/0; en la tercera semilla conservó el modelo anterior. Los checkpoints uniformes se eligieron en750/250/1250. No se evaluaron las versiones últimas en partidas finales ni se cambió la selección tras observar victorias.

La reducción de errores sobre el buffer no demuestra generalización. En partidas, los modelos prioritarios seleccionados eligieron más minas deducibles que los uniformes en las tres semillas: 163 vs159, 158 vs147 y162 vs161. Una hipótesis pendiente es el desajuste entre selección por pérdida y desempeño autónomo; otra es la distribución de entrenamiento inducida por este porcentaje de prioridad. La prueba no las separa causalmente.

Siguiente propuesta: comparar selección de checkpoints por desempeño autónomo en un conjunto de validación separado, conservando una prueba final nueva. No se inició esa fase ni se sustituyó el agente servido.

## Métricas en partidas

```json\n{json.dumps(metrics,indent=2)}\n```

Las políticas recorren posiciones distintas: oportunidades seguras y longitud de partidas pueden cambiar. Los conteos de minas seleccionadas no se interpretan como tasas por clic.

## Protocolo

Mismos datos y presupuesto entre brazos, sin nuevas trayectorias de entrenamiento. Se usan únicamente las rondas de experiencia disponibles para el checkpoint ampliado seleccionado anteriormente. Pesos, Adam y RNG restaurados. Cerebro, encoder, mapa óptico, arquitectura y pérdida uniformemente distribuida sobre jugadas seguras permanecen iguales.

1,500 updates por brazo/semilla, batch 64, Adam lr 0.001, clipping 5. Uniforme: 32 posiciones originales equilibradas por tamaño/categoría +32 uniformes del buffer DAgger. Prioritario: mismas 32 originales +16 uniformes del buffer +16 errores actuales donde el argmax legal elige una mina certificada pese a haber acción segura. Si no hay errores, se vuelve a 32 uniformes. Se recalculan errores cada 250; no hay corrección por importancia: se cambia deliberadamente la distribución de entrenamiento. Certificados obtenidos exclusivamente de observaciones públicas; no se filtran minas durante inferencia.

Selección por pérdida mínima en el holdout original cada 250, incluyendo checkpoint inicial. Los checkpoints finales no se eligen por victorias. Benchmark de 500 layouts nuevos 7×7/7 minas, reservado antes de entrenar, excluido de todos los datasets/benchmarks/colección anteriores. Apertura central segura idéntica. Evaluación autónoma sin maestro.

La prueba del pool de prioridad pasó. Auditoría independiente verificó 34,637 filas de certificados públicos, pools iniciales exactos, separación del benchmark, continuidad de contadores de Adam, doce pares de restauración con siguiente actualización exacta y hashes originales intactos. No se sustituyó agente servido. Fuentes, hashes, certificados por casilla, listas de errores por actualización, checkpoints con Adam/RNG y resultados por partida conservados. Las métricas de ajuste no demuestran generalización ni ventaja anatómica; el contraste importante es con muestreo uniforme a igual presupuesto.
'''
 Path('research/RESULTADO_PRIORIZACION_MINAS.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
