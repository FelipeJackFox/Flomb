"""Conditional paired analysis across head seeds on one shared unseen benchmark."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.validate_spatial_decoder import OUT,SEEDS


def main():
 assert (OUT/'completed.json').exists()
 names=['original',*[f'{v}-{s}' for s in SEEDS for v in ('pointwise','spatial')]]
 rows={n:json.loads((OUT/f'{n}-games.json').read_text()) for n in names}
 keys=[(r['seed'],r['layout_hash']) for r in rows['original']]
 for n in names:assert [(r['seed'],r['layout_hash']) for r in rows[n]]==keys
 play=np.array([not r['automatic'] for r in rows['original']]);n=int(play.sum())
 wins={k:np.array([r['won'] for r in v],float)[play] for k,v in rows.items()}
 rng=np.random.default_rng(20261005)
 compare={}
 for control in ('original','pointwise'):
  pairs=[]
  for seed in SEEDS:
   a=wins['original' if control=='original' else f'pointwise-{seed}'];b=wins[f'spatial-{seed}']
   plus=int(((b==1)&(a==0)).sum());minus=int(((a==1)&(b==0)).sum())
   pairs.append(dict(seed=seed,spatial_only=plus,control_only=minus,difference_pp=float(100*(b-a).mean()),mcnemar_exact_p=float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.))
  delta=np.mean([wins[f'spatial-{s}']-wins['original' if control=='original' else f'pointwise-{s}'] for s in SEEDS],axis=0)
  samples=delta[rng.integers(n,size=(10000,n))].mean(1)*100
  compare[control]=dict(pairs=pairs,mean_difference_pp=float(delta.mean()*100),conditional_board_bootstrap_95_pp=np.quantile(samples,[.025,.975]).tolist())
 totals={k:dict(wins=int(v.sum()),n=n,rate=float(v.mean())) for k,v in wins.items()}
 spatial=np.array([totals[f'spatial-{s}']['rate'] for s in SEEDS]);pointwise=np.array([totals[f'pointwise-{s}']['rate'] for s in SEEDS])
 fig,ax=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
 ax[0].axhline(totals['original']['rate']*100,color='#737f8c',linestyle='--',label='Original congelado')
 for i,(label,val,color) in enumerate([('Puntual',pointwise,'#d97706'),('Espacial',spatial,'#087f8c')]):
  x=np.arange(3)+(i-.5)*.26;ax[0].bar(x,val*100,.25,color=color,label=label)
  for xx,v in zip(x,val):ax[0].text(xx,v*100+.3,f'{round(v*n)}/{n}',ha='center',fontsize=8)
 ax[0].set_xticks(range(3),[str(s) for s in SEEDS]);ax[0].set(xlabel='Semilla del lector',ylabel='Victorias autónomas (%)',title='Mismos tableros, tres semillas');ax[0].set_ylim(0,10);ax[0].legend(fontsize=8,loc='upper left')
 for i,control in enumerate(('original','pointwise')):
  d=compare[control];value=d['mean_difference_pp'];lo,hi=d['conditional_board_bootstrap_95_pp'];ax[1].errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='o',capsize=6,color='#087f8c')
  ax[1].text(i,hi+.2,f'{value:+.2f} pp',ha='center')
 ax[1].set_xlim(-.35,1.35);ax[1].set_ylim(-.3,6.5);ax[1].axhline(0,color='gray',linestyle=':');ax[1].set_xticks([0,1],['Contra original','Contra puntual']);ax[1].set(ylabel='Diferencia de victorias (puntos porcentuales)',title='Promedio de lectores · IC por tableros')
 fig.savefig('research/spatial-consistency-results.png',dpi=160);plt.close(fig)
 table=[]
 for seed in SEEDS:
  a=totals[f'pointwise-{seed}'];b=totals[f'spatial-{seed}'];table.append(f"| {seed} | {a['wins']}/{n} ({a['rate']*100:.1f}%) | {b['wins']}/{n} ({b['rate']*100:.1f}%) |")
 memo=json.loads((OUT/'verification.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());sel=json.loads((OUT/'selection.json').read_text())
 hitratio=memo['memo_hits']/(memo['memo_hits']+memo['memo_misses'])
 text=f'''# Consistencia del decoder espacial

Corrida `spatial-consistency-001`, terminada. Tres semillas de lector y un único cerebro congelado; 500 layouts nuevos de 7×7/7 minas, reservados antes de entrenar.

## Resultados autónomos

Original congelado: **{totals['original']['wins']}/{n} ({totals['original']['rate']*100:.1f}%)**. Se excluyen {500-n} victorias por apertura automática. Ningún maestro elige acciones.

| Semilla del lector | Puntual (capacidad comparable) | Espacial |
|---|---|---|
{chr(10).join(table)}

Promedios entre tres lectores: puntual {pointwise.mean()*100:.2f}%, espacial {spatial.mean()*100:.2f}%. El espacial supera al puntual en {int((spatial>pointwise).sum())}/3 semillas y al original en {int((spatial>totals['original']['rate']).sum())}/3.

![Resultados](spatial-consistency-results.png)

## Comparación emparejada

Cada política juega los mismos tableros. `spatial_only` indica partidas que gana solo el espacial; `control_only`, solo el comparador. Diferencia media y percentiles de 10,000 remuestreos emparejados de tableros:

```json
{json.dumps(compare,indent=2)}
```

Los intervalos quedan **condicionados a los tres lectores entrenados**. No cuantifican incertidumbre entre todos los posibles entrenamientos o cerebros. Son 500 layouts compartidos, no 1,500 independientes. Los p-valores exactos por semilla son exploratorios y no están corregidos por comparaciones múltiples.

## Qué se mantuvo fijo

Arquitecturas del piloto (~12.8k parámetros cada una), 1,000 updates, batch64, Adam lr0.001, clipping5, minibatches balanceados por tamaño/categoría. Las semillas20261003/4 se entrenaron desde cero; la20261002 se reutilizó sin cambios. Selección por pérdida de validación cada100 updates; no se eligió una semilla por sus partidas finales.

Mismo checkpoint cerebral retina_plastic-20260926, congelado junto con encoder, ganancias y mapa óptico. Mismas 10,000 posiciones train y 1,000 valid. El único camino de las pistas del tablero hacia el lector pasa por la actividad neuronal. Contexto público: tamaño/cantidad de minas. Esto verifica consistencia del **lector**, no tres entrenamientos completos del cerebro.

Los 500 layouts se excluyeron de los datasets previos y listas de partidas guardadas, incluidos los del piloto espacial. 5×5 no se volvió a presentar como prueba inédita porque su universo de layouts quedó agotado.

## Optimización y verificación

Se reutilizaron cachés de entrenamiento inmutables. En inferencia se memoriza únicamente el mapa neuronal determinista, mediante hash de observación pública y contexto. Los lectores no comparten decisiones ni labels. Aciertos de caché: {memo['memo_hits']}; mapas nuevos: {memo['memo_misses']}; reutilización {hitratio*100:.1f}%. No se infiere de este porcentaje un speedup igual: también hay costes del entorno, solver diagnóstico y cabeza.

La caché y el lector original se comprobaron contra forward original en un batch mixto5×5/7×7 y al reordenarlo; resultados compatibles dentro de tolerancia y reordenado exacto. Hashes de cachés, dataset y checkpoints originales comprobados al finalizar. Fuentes, manifest, selección, pesos y resultados por partida preservados en la carpeta de corrida.

Selección por validación:
```json
{json.dumps(sel,indent=2)}
```

## Alcance de la decisión

Esta prueba permite decidir si seguir investigando lectura espacial bajo este entrenamiento. No demuestra que el conectoma supere una CNN convencional, ni generalización a16×16, ni que el cerebro congelado esté adquiriendo estrategias. El nivel absoluto de victorias debe valorarse junto con la mejora relativa.

No se modificó el agente servido ni se desplegó el modelo automáticamente. Mantener este benchmark como evaluación final; no ajustar repetidamente arquitectura o hiperparámetros mirando sus resultados.

Reproducir: `.venv/bin/python -m experiments.validate_spatial_decoder` (rechaza sobrescritura) y `.venv/bin/python -m experiments.report_spatial_consistency`. Requiere la caché y checkpoint del piloto, cuyos hashes están en manifest.json.
'''
 Path('research/RESULTADO_CONSISTENCIA_ESPACIAL.md').write_text(text)
 write=dict(totals=totals,comparisons=compare,spatial_mean=float(spatial.mean()),pointwise_mean=float(pointwise.mean()))
 (OUT/'summary.json').write_text(json.dumps(write,indent=2));print(json.dumps(write,indent=2))

if __name__=='__main__':main()
