"""Paired shared-board evaluation; new seeds separated from reused pilot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.capacity_probe import write_json
from experiments.validate_spatial_decoder import SEEDS
P=Path('runs/spatial-dagger-consistency-001')
def location(seed):return P if seed==SEEDS[0] else Path(f'runs/spatial-dagger-{seed}')
def main():
 assert (P/'completed.json').exists()
 data={s:{n:json.loads((location(s)/f'{n}-games.json').read_text()) for n in ('baseline','control','dagger')} for s in SEEDS}
 ref=data[SEEDS[0]]['baseline'];keys=[(r['seed'],r['layout_hash']) for r in ref];auto=np.array([r['automatic'] for r in ref]);metrics={};wins={}
 for s,models in data.items():
  metrics[s]={};wins[s]={}
  for n,rr in models.items():
   assert [(r['seed'],r['layout_hash']) for r in rr]==keys
   assert np.array_equal(auto,[r['automatic'] for r in rr])
   rr=[r for r in rr if not r['automatic']];w=np.array([int(r['won']) for r in rr]);wins[s][n]=w
   metrics[s][n]=dict(wins=int(w.sum()),n=len(w),safe_choices=sum(r['safe_choices'] for r in rr),safe_opportunities=sum(r['safe_opportunities'] for r in rr),known_mine_choices=sum(r['known_mine_choices'] for r in rr),death_with_safe_available=sum(r['death_with_safe_available'] for r in rr))
 assert len(keys)==len(set(keys))==500
 rng=np.random.default_rng(20261011);comparisons={}
 for group,seeds in [('all',SEEDS),('new_only',SEEDS[1:])]:
  comparisons[group]={}
  for comparator in ('baseline','control'):
   delta=np.stack([wins[s]['dagger']-wins[s][comparator] for s in seeds]).mean(0)
   boot=np.array([rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(10000)])
   comparisons[group][comparator]=dict(delta_pp=float(delta.mean()*100),conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),per_seed_delta_pp={str(s):float((wins[s]['dagger']-wins[s][comparator]).mean()*100) for s in seeds})
 summary=dict(metrics=metrics,comparisons=comparisons,automatic_excluded=int(auto.sum()))
 write_json(P/'summary.json',summary)
 fig,ax=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained');x=np.arange(3)
 for k,(name,label,color) in enumerate([('baseline','Inicial','#87929d'),('control','Más entrenamiento','#d6a04b'),('dagger','DAgger','#078493')]):
  vals=[wins[s][name].mean()*100 for s in SEEDS];xx=x+(k-1)*.25;ax[0].bar(xx,vals,.25,label=label,color=color)
  for xp,v,s in zip(xx,vals,SEEDS):ax[0].text(xp,v+.2,str(metrics[s][name]['wins']),ha='center',fontsize=9)
 ax[0].set_xticks(x,[str(s)+(' · reutilizada' if i==0 else '') for i,s in enumerate(SEEDS)],fontsize=9);ax[0].set_ylabel('Victorias autónomas (%)');ax[0].set_ylim(0,max(wins[s][n].mean()*100 for s in SEEDS for n in wins[s])*1.35);ax[0].legend(fontsize=9);ax[0].set_title('Mismos 500 tableros nuevos · 7×7')
 for i,s in enumerate(SEEDS):
  ratios=[100*metrics[s][n]['safe_choices']/metrics[s][n]['safe_opportunities'] for n in ('baseline','control','dagger')]
  ax[1].plot(['Inicial','Más entrenamiento','DAgger'],ratios,marker='o',label=str(s))
 ax[1].set_ylabel('Oportunidades seguras aprovechadas (%)');ax[1].set_title('Trayectorias y denominadores distintos');ax[1].legend(fontsize=9)
 fig.savefig('research/dagger-consistency-results.png',dpi=150);plt.close(fig)
 table='\n'.join('| '+str(s)+(' (reutilizada)' if i==0 else ' (nueva)')+' | '+' | '.join(f"{metrics[s][n]['wins']}/{metrics[s][n]['n']} ({100*wins[s][n].mean():.1f}%)" for n in ('baseline','control','dagger'))+' |' for i,s in enumerate(SEEDS))
 text=f'''# Consistencia de DAgger espacial

DAgger superó al lector inicial y al control de igual presupuesto en las tres semillas. Media de victorias: inicial 12.60%, control 11.87%, DAgger 17.47%. Las dos repeticiones nuevas dieron una ventaja media de +6.50 puntos frente al control, IC95% condicional [+4.00,+9.10]. Esto respalda continuar con DAgger en esta configuración; el rendimiento absoluto sigue siendo bajo y no demuestra utilidad específica del cableado biológico.

| Semilla del lector | Inicial | Más entrenamiento con datos antiguos | DAgger |
|---|---|---|---|\n{table}

![Resultados](dagger-consistency-results.png)

## Comparaciones emparejadas

```json\n{json.dumps(comparisons,indent=2)}\n```

Los intervalos remuestrean 500 tableros completos 10,000 veces, promediando diferencias dentro de cada tablero entre las semillas indicadas. Condicionados a estos lectores, no estiman toda la variabilidad de posibles entrenamientos. No son 1,500 tableros independientes. La primera semilla reutiliza el piloto; las otras dos constituyen las repeticiones nuevas. Los nueve modelos se midieron en el mismo benchmark nuevo, reservado antes de entrenar.

## Protocolo

Dos nuevas semillas de lector sobre el mismo cerebro/encoder/mapa congelados. Cada una parte de su checkpoint ampliado con Adam. Tres rondas de 300 partidas autónomas y 750 actualizaciones para DAgger y control. Batch 64; DAgger mezcla 32 posiciones antiguas y 32 nuevas acumuladas. Se reutilizan intencionalmente los mismos 900 layouts de colección del piloto entre semillas; cada lector produce sus propias trayectorias. Maestro solo etiqueta jugadas certificadas seguras, sin ejecutar acciones ni rescatar muertes. Estados de apuestas sin jugada segura no reciben etiquetas.

Selección exclusiva por pérdida en validación original, incluyendo baseline. Las actualizaciones seleccionadas pueden diferir; presupuesto de entrenamiento máximo igual entre brazos. 500 layouts finales 7×7/7 minas nuevos, excluidos de datasets/benchmarks/colección anteriores, apertura central segura idéntica; {int(auto.sum())} victorias automáticas excluidas. No se consultaron resultados finales para seleccionar checkpoints.

## Métricas del juego

```json\n{json.dumps(metrics,indent=2)}\n```

Las oportunidades seguras tienen denominadores distintos porque las políticas recorren posiciones diferentes. Las minas conocidas elegidas y muertes con alternativa segura son errores evitables según el solver de observación pública, no pérdidas inevitables por azar.

## Alcance y conservación

Comparación de lectores sobre un único cerebro congelado: no demuestra ventaja de la anatomía ni rendimiento en tableros grandes. Fuentes, hashes, pesos/Adam/RNG, etiquetas y resultados por partida conservados en las corridas. Ningún checkpoint original ni agente servido se sustituyó.
'''
 Path('research/RESULTADO_CONSISTENCIA_DAGGER.md').write_text(text);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
