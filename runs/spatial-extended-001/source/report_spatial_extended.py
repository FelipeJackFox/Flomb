"""Compare extended training against its paired 1000-update checkpoint."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.extend_spatial_decoder import OUT,SEEDS


def main():
 assert (OUT/'completed.json').exists()
 selection=json.loads((OUT/'selection.json').read_text());results=json.loads((OUT/'results.json').read_text())
 values={};keys=None;automatic=None
 for seed in SEEDS:
  for kind in ('baseline','extended'):
   rows=json.loads((OUT/f'{kind}-{seed}-games.json').read_text());current=[(r['seed'],r['layout_hash']) for r in rows]
   if keys is None:keys=current;automatic=np.array([r['automatic'] for r in rows])
   assert keys==current
   values[f'{kind}-{seed}']=np.array([r['won'] for r in rows],float)[~automatic]
 n=len(next(iter(values.values())));pairs=[]
 for seed in SEEDS:
  a,b=[values[f'{k}-{seed}'] for k in ('baseline','extended')]
  pairs.append(dict(seed=seed,before=int(a.sum()),after=int(b.sum()),n=n,difference_pp=float((b-a).mean()*100),extended_only=int(((b==1)&(a==0)).sum()),baseline_only=int(((a==1)&(b==0)).sum())))
 delta=np.mean([values[f'extended-{s}']-values[f'baseline-{s}'] for s in SEEDS],axis=0)
 rng=np.random.default_rng(20261006);boot=delta[rng.integers(n,size=(10000,n))].mean(1)*100;interval=np.quantile(boot,[.025,.975]).tolist()
 before=np.mean([values[f'baseline-{s}'].mean() for s in SEEDS]);after=np.mean([values[f'extended-{s}'].mean() for s in SEEDS])
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for i,(kind,label,color) in enumerate([('baseline','1,000 updates','#7b8794'),('extended','Ampliado, elegido por validación','#087f8c')]):
  x=np.arange(3)+(i-.5)*.28;y=np.array([values[f'{kind}-{s}'].mean()*100 for s in SEEDS]);ax[0].bar(x,y,.27,color=color,label=label)
  for xx,yy in zip(x,y):ax[0].text(xx,yy+.3,f'{round(yy*n/100)}/{n}',ha='center',fontsize=8)
 ymax=max(v.mean()*100 for v in values.values());ax[0].set_ylim(0,max(15,ymax*1.4));ax[0].legend(fontsize=8,loc='upper left');ax[0].set_xticks(range(3),[str(s) for s in SEEDS]);ax[0].set(xlabel='Semilla del lector',ylabel='Victorias autónomas (%)',title='Mismos 500 tableros inéditos de 7×7')
 for seed in SEEDS:
  h=json.loads((OUT/f'history-{seed}.json').read_text());ax[1].plot([r['step'] for r in h],[r['validation_loss'] for r in h],label=str(seed))
 ax[1].set(xlabel='Actualizaciones totales',ylabel='Pérdida de validación',title='Cerebro congelado · selección por validación');ax[1].legend(fontsize=8)
 fig.savefig('research/spatial-extended-results.png',dpi=160);plt.close(fig)
 lines=[]
 for p,s in zip(pairs,selection):lines.append(f"| {p['seed']} | {s['chosen_step']} / {s['stopped_step']} | {p['before']}/{n} ({100*p['before']/n:.1f}%) | {p['after']}/{n} ({100*p['after']/n:.1f}%) | {p['difference_pp']:+.1f} pp |")
 improved=sum(p['after']>p['before'] for p in pairs)
 text=f'''# Ampliación del lector espacial con cerebro congelado

Corrida `spatial-extended-001`, terminada. Se compararon tres continuaciones contra sus respectivos checkpoints de 1,000 actualizaciones en un benchmark nuevo reservado antes de entrenar.

## Resultado autónomo

Promedio entre lectores: **{before*100:.2f}% → {after*100:.2f}%**, diferencia **{delta.mean()*100:+.2f} puntos porcentuales**. Mejoró nominalmente en {improved}/3 semillas. Intervalo 95% por bootstrap emparejado de tableros: [{interval[0]:+.2f}, {interval[1]:+.2f}] pp, condicionado a estos tres lectores.

| Semilla | Update elegido / realizado | Antes (1,000) | Después | Diferencia |
|---|---|---|---|---|
{chr(10).join(lines)}

![Resultados](spatial-extended-results.png)

500 layouts 7×7/7 minas nuevos y compartidos por los seis modelos, excluidos de datasets y benchmarks anteriores. {int(automatic.sum())} aperturas automáticas excluidas; {n} partidas con decisiones. Argmax legal, sin intervención del maestro. Tres lectores sobre los mismos tableros no son 1,500 layouts independientes.

Comparación emparejada por semilla:
```json
{json.dumps(pairs,indent=2)}
```

`extended_only`: gana exclusivamente el ampliado; `baseline_only`: exclusivamente el checkpoint de 1,000. El intervalo se calculó con 10,000 remuestreos de tableros (semilla 20261006), promediando primero las diferencias de los tres lectores dentro de cada tablero; no estima toda la variabilidad entre posibles entrenamientos.

## Protocolo y conservación del estado

Mismo decoder espacial de 12,769 parámetros, mismo encoder/cerebro congelados, mismas 10,000 posiciones reales y 1,000 de validación. Batch 64, Adam lr 0.001, clipping 5, minibatches equilibrados por tamaño/categoría. Máximo 5,000 updates totales. Evaluación de validación cada 250 después de 1,000; parada tras seis comprobaciones sin mejorar al menos 0.0001 respecto al ancla anterior. El mejor checkpoint puede ser el original de 1,000: nunca se fuerza adoptar el último.

Los checkpoints previos no contenían Adam. Para continuar sin reiniciar el optimizador, se reprodujeron exactamente las primeras 1,000 actualizaciones para cada semilla. Se exigió igualdad bit a bit de todos los pesos con el checkpoint anterior antes de avanzar. Las tres reproducciones pasaron. Ahora se guardan último/mejor checkpoint con Adam, generador y número de actualización.

Selección y duración (incluye reconstrucción de 1,000 updates):
```json
{json.dumps(selection,indent=2)}
```

No se eligieron hiperparámetros, checkpoints ni semillas consultando los resultados finales de juego. Benchmarks anteriores permanecieron fuera de esta selección. Las cachés reducen la fase de entrenamiento a trabajar únicamente sobre actividad neuronal; inferencia usa memoización previamente verificada, sin compartir acciones entre lectores.

## Alcance

Se mide presupuesto adicional del lector sobre un único checkpoint cerebral. El cerebro congelado no aprende nuevos pesos, y la prueba no demuestra superioridad frente a una CNN, rendimiento en 16×16 ni ventaja del cableado biológico. El nivel absoluto de victorias sigue siendo el criterio práctico, además de la mejora relativa.

Archivos originales verificados por SHA256 sin cambios. No se sustituyó el agente servido ni se lanzó otra fase automáticamente. Fuente/checkpoints/historias/resultados por partida en runs/spatial-extended-001/. Reproducción: `.venv/bin/python -m experiments.extend_spatial_decoder` (protege corrida existente) y `.venv/bin/python -m experiments.report_spatial_extended`.
'''
 Path('research/RESULTADO_ESPACIAL_AMPLIADO.md').write_text(text)
 summary=dict(before_mean=float(before),after_mean=float(after),difference_pp=float(delta.mean()*100),conditional_board_bootstrap95_pp=interval,pairs=pairs)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
