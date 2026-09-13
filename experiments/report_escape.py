"""Generate the functional diagnostic report from measured results."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.escape_functional import OUT,ROOT,REF

r=json.loads((OUT/'results.json').read_text());br=json.loads((OUT/'brian2_reference.json').read_text())
def pick(model,kind='looming',sil=()):return [x for x in r if x['model']==model and x['condition']==kind and x['silenced']==list(sil)]
# Scientific controls: identical external events for matched seed, no bypass,
# independent Brian2 implementation agrees on presence/absence of response.
for seed in range(3):
 a=pick('lif_reference')[seed];c=pick('lif_reference',sil=('LC4','LPLC2'))[seed]
 assert a['input_spikes']==c['input_spikes'] and a['dnp01_spikes']>0 and c['dnp01_spikes']==0
assert br[0]['dnp01_spikes']>0 and br[1]['dnp01_spikes']==0
assert all(x['dnp01_spikes']==0 for x in pick('lif_normalized'))
for cycles in (3,6,12):
 assert pick(f'rate_{cycles}')[0]['peak_activation']>0
 assert pick(f'rate_{cycles}',sil=('LC4','LPLC2'))[0]['peak_activation']==0
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
t=np.arange(5,305,5)
ax[0].plot(t,pick('lif_reference')[0]['trace_hz'],label='LIF comunitario · semilla 0',color='#087f8c')
ax[0].plot(t,br[0]['trace_hz'],label='Brian2 · semilla 0',alpha=.8,color='#d97706')
ax[0].set(title='Respuesta temporal: no hay paridad exacta',xlabel='Tiempo simulado (ms)',ylabel='DNp01, Hz en ventanas de 5 ms');ax[0].legend(fontsize=8)
for cycles in (3,6,12):ax[1].plot(t,pick(f'rate_{cycles}')[0]['trace_activation'],label=f'{cycles} ciclos')
ax[1].set(title='Nuestra dinámica también transmite señal',xlabel='Tiempo del estímulo (ms)',ylabel='Activación tanh media (no Hz)');ax[1].legend()
labels=['Normal','Sin LC4','Sin LPLC2','Sin ambas','LIF normalizado'];groups=[pick('lif_reference',sil=s) for s in [(),('LC4',),('LPLC2',),('LC4','LPLC2')]]+[pick('lif_normalized')]
for i,g in enumerate(groups):
 vals=[x['dnp01_spikes'] for x in g];ax[2].bar(i,np.mean(vals),color='#087f8c' if i<4 else '#d97706');ax[2].scatter(np.repeat(i,len(vals)),vals,color='black',s=14,zorder=3)
ax[2].set_xticks(range(5),labels,rotation=25,ha='right');ax[2].set(title='Controles del LIF · tres semillas',ylabel='Disparos DNp01 en 300 ms')
fig.suptitle('Circuito de escape · mismo grafo FlyWire · sin entrenamiento',fontsize=15)
plot=ROOT/'research/escape-functional.png';fig.savefig(plot,dpi=160);plt.close(fig)
rows=[]
for name,g in zip(labels,groups):rows.append('| '+name+' | '+', '.join(str(x['dnp01_spikes']) for x in g)+' |')
rate=[pick(f'rate_{c}')[0]['peak_activation'] for c in (3,6,12)]
source=OUT/'source';source.mkdir(exist_ok=True)
for name in ['escape_functional.py','escape_brian_reference.py','report_escape.py']:shutil.copy2(ROOT/'experiments'/name,source/name)
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
(OUT/'verification.json').write_text(json.dumps(dict(paired_input_controls=True,rate_lesion_controls=True,brian2_functional_replication=True,numerical_parity=False,source_sha256=checks),indent=2))
text=f'''# Circuito de escape: comparación funcional

13 de septiembre de 2026. Experimento terminado, sin entrenamiento ni cambios a checkpoints de Buscaminas.

## Resultado que cambia el diagnóstico

La dinámica simplificada **sí transmite actividad hacia DNp01 por el circuito conocido**. La respuesta desaparece al silenciar LC4 y LPLC2. Por tanto, este experimento no respalda que usar tanh o tres ciclos destruya necesariamente este circuito. No demuestra que nuestra red pueda aprender las relaciones de Buscaminas.

![Resultados](escape-functional.png)

## Diseño y procedencia

Mismo grafo FlyWire v783 exportado por Fruit Fly Laboratory: 139,255 nodos y 3,732,460 conexiones. Commit `26672e06427c12c61536ce1bd93dae7442944681`; hashes verificados contra su manifiesto. Neuronas de entrada/salida comprobadas contra sus tipos e IDs en el archivo de neuronas. **Copia comunitaria: no se reconstruyó ni verificó independientemente contra todos los archivos originales de FlyWire.** No es MaleCNS y no se trasplantaron nuestros checkpoints.

Fuente: [repositorio fijado](https://github.com/vaibhavkedarisetti/fruit-fly-lab/tree/26672e06427c12c61536ce1bd93dae7442944681). Modelo publicado: [Shiu, código y parámetros](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py).

Objeto con semitamaño 5 mm a 50 mm, azimut 45 grados, inicio 20 ms, velocidad 250 mm/s; desaparece al alcanzar distancia cero. Duración 300 ms, paso LIF 0.1 ms. Campos receptivos redondeados a tres decimales del export web; encoder de tamaño/expansión adaptado del repositorio, no visión de píxeles. Tres semillas 0/1/2; misma estimulación aleatoria para intervenciones emparejadas. El conteo es la suma de las dos neuronas DNp01.

La selectividad a acercamiento frente a alejamiento/objeto quieto está **introducida en el encoder**: estos dos controles reciben cero estímulo. No prueba que el conectoma descubra por sí solo la dirección del movimiento. Las lesiones con entrada idéntica sí prueban dependencia de propagación interna.

## LIF comunitario: mediciones locales

| Condición | Disparos DNp01, semillas 0/1/2 |
|---|---|
{chr(10).join(rows)}

Objeto quieto y alejándose: cero disparos, como se espera de sus entradas nulas. El lector nunca consulta la geometría para decidir si hay salida. No se simuló un cuerpo ni una partida de Dinosaurio: se midió el circuito neuronal.

## Dinámica local de tasas sobre el mismo grafo

Se reprodujeron normalización por suma absoluta entrante, ganancias unitarias, estado inicial tanh(entrada) y actualización tanh(W·estado + 0.15·entrada). Entrada = tasa sensorial / 150; es una conversión de ingeniería. Se reinicia estado por observación, como RetinaPolicy. No usamos encoder/decoder entrenados de Buscaminas, que esperan otro problema y otro grafo.

| Ciclos | Pico medio DNp01, unidades tanh |
|---|---:|
| 3 | {rate[0]:.8f} |
| 6 | {rate[1]:.8f} |
| 12 | {rate[2]:.8f} |

El aumento de 3 a 12 ciclos es {100*(rate[2]/rate[0]-1):.2f}%. Con ambas poblaciones cortadas, cero en las tres variantes. La amplitud tanh no es una frecuencia de disparo y no se compara numéricamente con Hz. Una salida pequeña tampoco implica por sí sola incapacidad: el decoder puede amplificarla.

## Verificación independiente con Brian2

Se ejecutaron las ecuaciones, umbrales, retardos y refractariedad publicados mediante Brian2 {br[0]['version']}, sobre el grafo completo y los mismos eventos externos de la semilla 0. Entrada en fase `synapses`, correspondiente a PoissonInput; el motor comunitario la aplica antes de comprobar umbral.

Brian2: **{br[0]['dnp01_spikes']} disparos**, primer disparo {br[0]['first_spike_ms']} ms; al cortar ambas entradas, **{br[1]['dnp01_spikes']}**. Motor comunitario: **{pick('lif_reference')[0]['dnp01_spikes']} disparos**, primer disparo {pick('lif_reference')[0]['first_spike_ms']} ms. Hay replicación funcional, **no paridad numérica**. La planificación temporal y precisión son diferencias posibles; no aislamos toda la causa. No adoptarlo como implementación equivalente a Brian2 sin resolverlo.

## Qué demuestra el control de normalización

Aplicar la normalización entrante a los pesos del LIF, conservando el factor sináptico 0.275 mV y todos los demás parámetros, elimina los disparos de salida. Esto prueba que **no se pueden intercambiar escalas de pesos entre modelos sin recalibrar**. No prueba que la normalización sea mala para tanh: esa variante sí conserva señal.

## Decisión siguiente

El cuello de botella de Buscaminas sigue sin localizarse. Este circuito simple es un control positivo logrado, no un entrenamiento nuevo ni una validación de visión completa. No cambiar todo a LIF basándose en este ensayo.

Siguiente prueba recomendada: habilidades mínimas de Buscaminas (deducción de una pista), evaluación en configuraciones nuevas y lectura de información en entrada, capas intermedias y salida. Comparar contra CNN y grafo reconfigurado; comprobar en qué etapa deja de ser decodificable la respuesta. Mantener separada la réplica biológica de la investigación de aprendizaje.

## Reproducibilidad

Comandos desde la raíz: `.venv/bin/python -m experiments.escape_functional`, `.venv/bin/python -m experiments.escape_brian_reference`, `.venv/bin/python -m experiments.report_escape`. El primer comando rechaza sobrescribir una corrida completada. Resultados y fuentes fijadas: `runs/escape-functional-001/`; registro de descargas: `reference/escape-functional/sources.json`. Dependencias añadidas: brian2==2.10.1, matplotlib==3.11.2. Los controles emparejados e independientes se verifican al generar este informe.
'''
(ROOT/'research/RESULTADO_CIRCUITO_ESCAPE.md').write_text(text)
print(text[:600]);print('Report and plot saved.')
