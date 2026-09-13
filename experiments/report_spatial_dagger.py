"""Paired final-game report for the single-seed DAgger pilot."""
import json,pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.capacity_probe import write_json
P=Path('runs/spatial-dagger-001')
def main():
 assert (P/'completed.json').exists()
 results=json.loads((P/'results.json').read_text());rows={n:json.loads((P/f'{n}-games.json').read_text()) for n in results}
 keys=None
 for name,rr in rows.items():
  kk=[(r['seed'],r['layout_hash']) for r in rr]
  if keys is None:keys=kk
  assert kk==keys and len(set(kk))==500
 auto=np.array([r['automatic'] for r in rows['baseline']]);assert all(np.array_equal(auto,[r['automatic'] for r in rr]) for rr in rows.values())
 metrics={}
 for name,rr in rows.items():
  rr=[r for r in rr if not r['automatic']]
  metrics[name]=dict(wins=sum(r['won'] for r in rr),n=len(rr),safe_choices=sum(r['safe_choices'] for r in rr),safe_opportunities=sum(r['safe_opportunities'] for r in rr),known_mine_choices=sum(r['known_mine_choices'] for r in rr),death_with_safe_available=sum(r['death_with_safe_available'] for r in rr),death_after_exact_half_min_risk=sum(r['death_after_exact_half_min_risk'] for r in rr),selected_step=results[name]['selected_step'])
 rng=np.random.default_rng(20261009);comparisons={}
 for control in ('baseline','control'):
  d=np.array([int(a['won'])-int(b['won']) for a,b in zip(rows['dagger'],rows[control])])[~auto]
  boot=np.array([rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)])
  comparisons[control]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),dagger_only=int((d==1).sum()),comparator_only=int((d==-1).sum()))
 summary=dict(metrics=metrics,comparisons=comparisons,automatic_excluded=int(auto.sum()))
 write_json(P/'summary.json',summary)
 hist=json.loads((P/'history.json').read_text());collections=[json.loads((P/f'collection-{i}.json').read_text()) for i in range(1,4)]
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 names=list(metrics);labels=['Sin cambios','Más entrenamiento','DAgger']
 vals=[100*metrics[n]['wins']/metrics[n]['n'] for n in names]
 ax[0].bar(labels,vals,color=['#87929d','#d6a04b','#078493']);ax[0].set_ylabel('Victorias autónomas (%)');ax[0].set_ylim(0,max(vals)*1.4+1);ax[0].set_title('Mismos 500 tableros nuevos · 7×7')
 for i,n in enumerate(names):ax[0].text(i,vals[i]+.3,f"{metrics[n]['wins']}/{metrics[n]['n']}",ha='center')
 for n,label in [('control','Datos antiguos'),('dagger','DAgger')]:
  h=[r for r in hist if r['variant']==n];ax[1].plot([r['step'] for r in h],[r['validation_loss'] for r in h],label=label)
 ax[1].set_title('Validación original · cerebro congelado');ax[1].set_xlabel('Actualizaciones adicionales');ax[1].set_ylabel('Pérdida');ax[1].legend()
 fig.savefig('research/spatial-dagger-results.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {label} | {metrics[n]['wins']}/{metrics[n]['n']} ({vals[i]:.1f}%) | {metrics[n]['safe_choices']}/{metrics[n]['safe_opportunities']} | {metrics[n]['known_mine_choices']} | {metrics[n]['death_with_safe_available']} | {metrics[n]['selected_step']} |" for i,(n,label) in enumerate(zip(names,labels)))
 text=f'''# DAgger sobre el lector espacial: piloto terminado

Un único lector, semilla 20261002, sobre el cerebro congelado. Tres rondas de 300 partidas de colección y 750 actualizaciones; 2,250 actualizaciones adicionales por variante. Se compara con el mismo checkpoint inicial y un control con el mismo presupuesto de actualizaciones sobre datos antiguos.

| Variante | Victorias | Jugadas seguras tomadas / disponibles | Minas conocidas elegidas | Muertes con alternativa segura | Update seleccionado |
|---|---|---|---|---|---|\n{table}

![Resultados](spatial-dagger-results.png)

DAgger menos baseline: {comparisons['baseline']['delta_pp']:+.2f} puntos porcentuales, IC95% emparejado {comparisons['baseline']['ci95_pp']}. DAgger menos control de actualizaciones: {comparisons['control']['delta_pp']:+.2f} pp, IC95% {comparisons['control']['ci95_pp']}. Bootstrap de 10,000 remuestreos de tableros, condicionado a este único entrenamiento.

En este piloto, DAgger mejora frente al checkpoint inicial. La diferencia frente al control con igual presupuesto tiene un intervalo que incluye cero: todavía no demuestra una ventaja atribuible a DAgger. El siguiente paso recomendado es repetir la comparación con las otras dos semillas y otro benchmark reservado, antes de escalar rondas o modificar el cerebro.

Los 500 layouts finales son nuevos, reservados antes de recoger datos; separados de los 900 layouts de colección y todos los datasets/benchmarks previos. Apertura central segura idéntica; se excluyeron {int(auto.sum())} victorias automáticas. Ninguna acción del maestro durante colección o evaluación. Las trayectorias de las variantes difieren: los denominadores de oportunidades seguras también.

## Recolección y selección

DAgger beta=0: el alumno decide todos los clics y termina si muere. El maestro etiqueta únicamente estados con alguna jugada certificada segura a partir de observación pública; no aporta el mapa de minas. Se incluyen aciertos y errores del alumno. Mitad del minibatch proviene de las posiciones originales equilibradas por tamaño/categoría y mitad de todas las posiciones acumuladas. No se enseña a resolver apuestas sin jugada segura en este piloto.

Colección por ronda:\n```json\n{json.dumps(collections,indent=2)}\n```

La selección usa exclusivamente la pérdida en las 1,000 posiciones originales de validación y permite conservar el checkpoint inicial (update 0). Los resultados finales de juego no intervienen en selección. El lector utilizado para recoger cada ronda es el estado más reciente, aunque la selección final elija un estado anterior.

## Verificación y alcance

Prueba del colector contra trayectorias independientes verifica que ejecuta las acciones del alumno, incluidos errores, y conserva etiquetas y tableros correctos. Cinco pruebas del colector, memoización y lector pasan. Checkpoints previos preservados mediante SHA256; los nuevos conservan pesos, Adam y RNG. Actividad neuronal cacheada sin compartir decisiones.

Es una prueba de adaptación del lector con un solo cerebro y una sola semilla, no evidencia de ventaja biológica ni generalización a tableros grandes. Los datos nuevos solo son 7×7; validar retención en otras dificultades requiere otra evaluación. Los contadores de muerte en apuestas exactas 50/50 están en summary.json y no se convierten en victorias ficticias. No se sustituyó el agente servido.
'''
 Path('research/RESULTADO_DAGGER_ESPACIAL.md').write_text(text)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
