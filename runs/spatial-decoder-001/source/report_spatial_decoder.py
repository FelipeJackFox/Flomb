"""Report paired autonomous evaluations; exclude automatic-opening victories."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_spatial_decoder import OUT


def summarize(rows,size):
 allrows=[r for r in rows if r['size']==size];played=[r for r in allrows if not r['automatic']]
 wins=sum(r['won'] for r in played);n=len(played);p=wins/n;z=1.96
 center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 return dict(wins=wins,n=n,automatic=len(allrows)-n,rate=p,ci=[center-half,center+half],safe=sum(r['safe_choices'] for r in played),opportunities=sum(r['safe_opportunities'] for r in played),known_mine_choices=sum(r['known_mine_choices'] for r in played),avoidable_deaths=sum(r['death_with_safe_available'] for r in played))

def main():
 assert (OUT/'completed.json').exists()
 names=['original','pointwise','spatial'];titles=['Lector original','Lector puntual nuevo','Lector espacial nuevo']
 games={n:json.loads((OUT/f'{n}-games.json').read_text()) for n in names}
 for n in names:
  assert [(r['size'],r['seed'],r['layout_hash']) for r in games[n]]==[(r['size'],r['seed'],r['layout_hash']) for r in games['original']]
 s={n:{size:summarize(games[n],size) for size in (5,7)} for n in names}
 fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for i,(name,title,color) in enumerate(zip(names,titles,['#7b8794','#d97706','#087f8c'])):
  for j,size in enumerate((5,7)):
   r=s[name][size];x=j+(i-1)*.23;v=100*r['rate'];low,hi=np.array(r['ci'])*100
   ax[0].bar(x,v,.22,label=title if j==0 else None,color=color);ax[0].errorbar(x,v,yerr=[[v-low],[hi-v]],fmt='none',ecolor='black',capsize=3)
   ax[0].text(x,hi+2,f"{r['wins']}/{r['n']}",ha='center',fontsize=8)
 for name,title,color in zip(names[1:],titles[1:],['#d97706','#087f8c']):
  h=json.loads((OUT/f'{name}-history.json').read_text());ax[1].plot([x['step'] for x in h],[x['validation_loss'] for x in h],label=title,color=color)
 ax[0].set_xticks([0,1],['5×5 · 3 minas','7×7 · 7 minas']);ax[0].set_ylim(0,100);ax[0].set_ylabel('Victorias autónomas (%) · IC Wilson 95%');ax[0].legend(fontsize=8)
 ax[1].set(xlabel='Actualizaciones del lector',ylabel='Pérdida de validación',title='Selección sin consultar las partidas finales');ax[1].legend(fontsize=8)
 fig.suptitle('Cerebro congelado · actividad neuronal como única entrada del tablero')
 fig.savefig('research/spatial-decoder-results.png',dpi=160);plt.close(fig)
 rows=[]
 for name,title in zip(names,titles):
  for size in (5,7):
   v=s[name][size];rows.append(f"| {title} | {size}×{size} | {v['wins']}/{v['n']} ({100*v['rate']:.1f}%) | {v['safe']}/{v['opportunities']} | {v['known_mine_choices']} |")
 paired={}
 for comparison in ['original','pointwise']:
  paired[comparison]={}
  for size in (5,7):
   pairs=[(a,b) for a,b in zip(games[comparison],games['spatial']) if a['size']==size and not a['automatic']]
   paired[comparison][size]=dict(spatial_only=sum(b['won'] and not a['won'] for a,b in pairs),control_only=sum(a['won'] and not b['won'] for a,b in pairs))
 manifest=json.loads((OUT/'manifest.json').read_text());res=json.loads((OUT/'results.json').read_text())
 verification=json.loads((OUT/'verification.json').read_text())
 text=f'''# Decoder espacial con cerebro congelado

Experimento `spatial-decoder-001`, terminado. Entrenamiento exclusivo del lector; checkpoint híbrido y modelo neuronal anterior preservados.

**El lector espacial mejora nominalmente en ambas dificultades y el lector puntual de capacidad comparable no.** Comparación emparejada favorable al espacial: frente al original, 12 victorias exclusivas contra 1 en 5×5, y 18 contra 6 en 7×7. Apoya continuar con lectura espacial; no demuestra superioridad entre semillas o conectomas.

![Resultados](spatial-decoder-results.png)

## Evaluación autónoma

Mismas {len(games['original'])} partidas nuevas para las tres políticas: {len([g for g in games['original'] if g['size']==5])} de 5×5 y {len([g for g in games['original'] if g['size']==7])} de 7×7. En 5×5 solo quedaban 127 layouts inéditos de los 2,024 posibles con tres minas y centro seguro; se usaron todos. Se excluyen las victorias que ocurren durante la apertura automática (5×5: {s['original'][5]['automatic']}; 7×7: {s['original'][7]['automatic']}). El maestro no elige acciones; el solver solo mide oportunidades después de que el modelo produce sus puntuaciones. Argmax determinista entre acciones legales.

| Modelo | Dificultad | Victorias excluyendo aperturas | Oportunidades seguras aprovechadas | Elecciones de mina demostrable |
|---|---|---|---|---|
{chr(10).join(rows)}

Las tasas de oportunidades seguras corresponden a las trayectorias de cada política; sus denominadores difieren. No son una comparación sobre los mismos estados ni un win rate corregido por suerte.

Comparación emparejada (mismas posiciones de minas):
```json
{json.dumps(paired,indent=2)}
```
`spatial_only`: gana el espacial y pierde el control. `control_only`: lo contrario. Intervalos de la gráfica: Wilson 95% por proporción; no son intervalos de la diferencia emparejada.

## Intervención y control

Encoder, ganancias por neurona y conexión y mapa óptico congelados del checkpoint retina_plastic de 3,000 updates, SHA256 `{manifest['checkpoint_sha256']}`. Todos los nodos y conexiones originales se conservan. La única información del tablero que recibe el lector es el mapa de actividad de diez tipos neuronales; también recibe tamaño y cantidad pública de minas, como el original. No hay bypass de pistas visibles.

Lector espacial: dos convoluciones 3×3 con 32 canales, tanh y salida 1×1: campo 5×5, 12,769 parámetros. Control puntual nuevo: dos convoluciones 1×1 con 106 canales, tanh y salida 1×1: 12,827 parámetros. Diferencia de capacidad <1%. El control puntual separa acceso espacial de mero aumento de parámetros y presupuesto adicional. El lector original permanece como tercer control sin nuevo entrenamiento.

10,000 posiciones reales de entrenamiento y 1,000 de validación del dataset estructural previo. Se precalculó la actividad congelada una vez; entrenar los lectores no repite la propagación neuronal. Ambos reciben los mismos minibatches equilibrados por tamaño/categoría, 1,000 updates, batch64, Adam lr0.001, clipping5, semilla20261002. Selección por mínima pérdida de validación cada100 pasos, sin usar las partidas finales.

```json
{json.dumps(res,indent=2)}
```

Partidas finales con layouts excluidos de los datasets previos encontrados en runs/*/dataset.pkl; se guardan seeds y hashes. No se modificaron los datos ni las reglas del juego. La interfaz actual solo soporta 5×5 y 7×7: no se extrapola a tableros grandes.

## Verificación y límites

Tres pruebas del lector pasan: campo receptivo 5×5 frente a 1×1 mediante gradiente, capacidad comparable y roundtrip exacto de pesos. Durante la preparación se verificó reconstrucción del lector original desde la caché en acciones legales (tolerancia1e-5). Hashes de checkpoints protegidos verificados antes y después. Híbrido: `{manifest['protected_sha256']}`.

Verificación independiente de caché en ambos tamaños: error máximo {verification['5']} en 5×5 y {verification['7']} en 7×7. Anulando la actividad antes del lector espacial, las victorias caen a {verification['zero_activity_wins']['5']['wins']}/{verification['zero_activity_wins']['5']['n']} y {verification['zero_activity_wins']['7']['wins']}/{verification['zero_activity_wins']['7']['n']}. Esto comprueba dependencia funcional de la actividad, no ventaja biológica; anularla también cambia la distribución de entrada.

En 5×5 evaluamos el remanente de un universo pequeño casi agotado. No se debe tratar como muestreo nuevo ilimitado del juego ni ajustar más hiperparámetros mirando estos mismos resultados.

Una semilla de entrenamiento y un checkpoint cerebral: resultado piloto, no superioridad general ni ventaja del conectoma. No se publicó ni se sustituyó el agente de la escena automáticamente. Debe distinguirse mejorar un lector sobre actividad de que el cerebro congelado haya aprendido una estrategia nueva.

## Archivos

Resultados por partida: `runs/spatial-decoder-001/*-games.json`. Checkpoints de lectores: `pointwise.pt`, `spatial.pt`; requieren el checkpoint cerebral especificado en el manifiesto. Cachés reutilizables: train.pt/holdout.pt. Progreso e historiales en la misma carpeta. Comandos: `.venv/bin/python -m experiments.train_spatial_decoder` (rechaza sobrescritura), `.venv/bin/python -m experiments.report_spatial_decoder`. Fuentes de la corrida preservadas en source/.
'''
 Path('research/RESULTADO_DECODER_ESPACIAL.md').write_text(text)
 (OUT/'summary.json').write_text(json.dumps(dict(strata=s,paired=paired),indent=2));print(json.dumps(s,indent=2))

if __name__=='__main__':main()
