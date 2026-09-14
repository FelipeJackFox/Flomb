"""Report paired experience expansion vs fixed-buffer continuation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.expand_dagger_experience import OUT,target,SEEDS
from experiments.capacity_probe import write_json

def main():
 assert (OUT/'completed.json').exists()
 models=('baseline','control','expanded');names=('DAgger anterior','Buffer fijo','Experiencia ampliada');metrics={};wins={};reference=None
 for seed in SEEDS:
  metrics[seed]={};wins[seed]={}
  for model in models:
   rr=json.loads((target(seed)/f'{model}-games.json').read_text());keys=[(r['seed'],r['layout_hash'],r['automatic']) for r in rr]
   if reference is None:reference=keys
   assert keys==reference and len(keys)==500 and len({k[1] for k in keys})==500
   rr=[r for r in rr if not r['automatic']];w=np.array([int(r['won']) for r in rr]);wins[seed][model]=w
   metrics[seed][model]=dict(wins=int(w.sum()),n=len(w),safe_choices=sum(r['safe_choices'] for r in rr),safe_opportunities=sum(r['safe_opportunities'] for r in rr),known_mine_choices=sum(r['known_mine_choices'] for r in rr),death_with_safe_available=sum(r['death_with_safe_available'] for r in rr),death_after_exact_half_min_risk=sum(r['death_after_exact_half_min_risk'] for r in rr))
 rng=np.random.default_rng(20261013);comparisons={}
 for other in ('baseline','control'):
  diffs=np.stack([wins[s]['expanded']-wins[s][other] for s in SEEDS]);delta=diffs.mean(0)
  boot=np.array([rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)])
  comparisons[other]=dict(mean_delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),per_seed_delta_pp={str(s):float(d.mean()*100) for s,d in zip(SEEDS,diffs)})
 means={m:float(np.mean([wins[s][m].mean() for s in SEEDS])*100) for m in models}
 selection=json.loads((OUT/'results.json').read_text());summary=dict(means_pct=means,comparisons=comparisons,metrics=metrics,selection=selection)
 write_json(OUT/'summary.json',summary)
 fig,ax=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained');x=np.arange(3)
 for i,(m,label,color) in enumerate(zip(models,names,('#87929d','#d6a04b','#078493'))):
  xx=x+(i-1)*.25;vv=[wins[s][m].mean()*100 for s in SEEDS];ax[0].bar(xx,vv,.25,label=label,color=color)
  for xp,v,s in zip(xx,vv,SEEDS):ax[0].text(xp,v+.2,str(metrics[s][m]['wins']),ha='center',fontsize=9)
 ax[0].set_xticks(x,[str(s) for s in SEEDS]);ax[0].set_ylim(0,max(wins[s][m].mean()*100 for s in SEEDS for m in models)*1.4);ax[0].set_ylabel('Victorias autónomas (%)');ax[0].set_title('Mismos 500 tableros nuevos · 7×7');ax[0].legend(fontsize=8)
 for s in SEEDS:
  v=[100*metrics[s][m]['safe_choices']/metrics[s][m]['safe_opportunities'] for m in models];ax[1].plot(['Anterior','Buffer fijo','Ampliado'],v,marker='o',label=str(s))
 ax[1].set_title('Trayectorias y denominadores distintos');ax[1].set_ylabel('Oportunidades seguras aprovechadas (%)');ax[1].legend(fontsize=9)
 fig.savefig('research/dagger-expanded-results.png',dpi=150);plt.close(fig)
 table='\n'.join('| '+str(s)+' | '+' | '.join(f"{metrics[s][m]['wins']}/{metrics[s][m]['n']} ({100*wins[s][m].mean():.1f}%)" for m in models)+' |' for s in SEEDS)
 text=f'''# Ampliación de experiencia DAgger

La ampliación ganó nominalmente más partidas en las tres semillas. Frente al DAgger anterior, la media subió +2.80 puntos (IC95% condicional [+0.93,+4.67]). Frente a continuar sobre la experiencia ya recogida, la ventaja fue +1.20 puntos (IC95% [−0.60,+3.00]): esta fase no establece un beneficio claro de recoger más datos frente a seguir entrenando con el buffer existente.

Los errores evitables persisten: clics en minas deducibles anterior→ampliado 170→176, 155→179 y 180→184 por cada 500 partidas. Las trayectorias se alargan y cambian; estos conteos no aíslan por sí solos la calidad por decisión. Las muertes con alternativa segura fueron 249→245, 224→245 y 243→241. Conviene diagnosticar esos errores antes de ampliar nuevamente el presupuesto.

| Semilla | DAgger anterior | Continuar con buffer fijo | Experiencia ampliada |
|---|---|---|---|\n{table}

Media entre lectores: anterior {means['baseline']:.2f}%, buffer fijo {means['control']:.2f}%, ampliado {means['expanded']:.2f}%.

![Resultados](dagger-expanded-results.png)

## Diferencias emparejadas

```json\n{json.dumps(comparisons,indent=2)}\n```

Bootstrap de 10,000 remuestreos por tablero, promedio de las diferencias entre lectores dentro de cada tablero. Intervalos condicionados a estas tres semillas; son 500 layouts compartidos, no 1,500 independientes. Un único cerebro congelado. Los modelos anteriores se volvieron a evaluar en este benchmark, sin comparar porcentajes de conjuntos distintos.

## Experimento

Tres rondas de 600 partidas nuevas y 750 actualizaciones por semilla. Baseline intacto: mejor DAgger anterior. Control y ampliado parten de ese mismo checkpoint con pesos, Adam y RNG. Batch 64: 32 posiciones originales equilibradas por tamaño/categoría y 32 de experiencia DAgger. El control mantiene fijo el buffer heredado; el ampliado agrega todas las nuevas posiciones recogidas. Ambos reciben 2,250 updates adicionales.

El maestro solo etiqueta posiciones con alguna jugada certificada segura desde observación pública. El lector ejecuta todas las acciones, incluidos errores, y termina al morir. No se etiquetan apuestas sin jugada segura ni se convierten pérdidas 50/50 en victorias. Apertura central segura automática igual para todos.

Selección por pérdida mínima en el holdout original, cada 250 updates, incluyendo baseline; el estado usado durante colección puede diferir del checkpoint finalmente elegido. Ningún resultado final de juego se usó para seleccionar. Nuevos 500 layouts finales y 1,800 de colección, excluidos de todos los datasets/benchmarks/colección previos; colección compartida entre lectores con trayectorias propias.

Selección por semilla:\n```json\n{json.dumps(selection,indent=2)}\n```

## Métricas específicas

```json\n{json.dumps(metrics,indent=2)}\n```

Las oportunidades seguras tienen denominadores distintos por trayectoria. Minas conocidas elegidas y muertes con alternativa segura son errores evitables según el solver público. El contador de muertes en apuestas exactas 50/50 se conserva separado.

Se recogieron 27,898 posiciones nuevas (9,648 / 9,220 / 9,030). Los checkpoints elegidos no usan necesariamente todas: las semillas segunda y tercera fueron seleccionadas durante la segunda ronda y sus terceros lotes no contribuyen a esos pesos. La experiencia nueva disponible al entrenar cada checkpoint elegido fue 9,648 / 5,925 / 5,904 posiciones, además de la heredada.

Cinco pruebas del colector/memo/lector pasan. Auditorías por semilla verificaron todas las etiquetas nuevas desde observación pública, conservación de las filas heredadas, separación de layouts, continuidad de contadores de Adam, doce pares de restauraciones con siguiente actualización exacta, pesos/estado finitos y hashes originales intactos. Las muestras de actividad cacheada (cuatro posiciones de cada ronda y semilla) coincidieron con el cálculo directo, error máximo 0. Esto es una comprobación muestreada, no una reejecución de toda la caché.

Fuentes, datasets, actividad por ronda, mejores/últimos checkpoints con Adam/RNG y hashes originales conservados. No se sustituyó agente servido. No se demuestra ventaja biológica ni generalización fuera de 7×7; para interpretar el beneficio de más datos, comparar con el control de buffer fijo además del baseline.
'''
 Path('research/RESULTADO_DAGGER_EXPERIENCIA_AMPLIADA.md').write_text(text);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
