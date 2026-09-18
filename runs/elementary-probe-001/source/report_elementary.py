import hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.elementary_probe import OUT

r=json.loads((OUT/'results.json').read_text());cnn=json.loads((OUT/'cnn_results.json').read_text());v=json.loads((OUT/'verification.json').read_text())
keys=['raw','encoder','sensory','cycle1','cycle2','cycle3','cycle3_target','neighborhood_pca10','decoder_hidden','logit']
labels=['Entrada visible','Encoder','Entrada neuronal','Ciclo 1, vecindad','Ciclo 2, vecindad','Ciclo 3, vecindad','Ciclo 3, solo casilla','Vecindad comprimida a 10','Decoder, capa oculta','Score de acción']
def mean(a):return np.mean([x['test_correct']/x['n']*100 for x in a])
fig,ax=plt.subplots(figsize=(11,6.8),layout='constrained');y=np.arange(len(keys))
for offset,(name,desc,col) in zip([-.18,.18],[('retina_plastic','Conectoma original','#087f8c'),('rewired_plastic','Grafo reconfigurado','#d97706')]):
 vals=[mean(r[name][k]) for k in keys];ax.barh(y+offset,vals,height=.34,label=desc,color=col)
 for yy,val in zip(y+offset,vals):ax.text(val+.5,yy,f'{val:.1f}%',va='center',fontsize=8)
ax.axvline(100/3,linestyle=':',color='gray',label='Azar: 33.3%');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,108)
ax.set_xlabel('Aciertos en 240 posiciones reservadas · media de 3 semillas del lector')
ax.set_title('Regla de una pista: la información es más accesible en la vecindad')
ax.legend(loc='lower right',fontsize=9);ax.spines[['top','right']].set_visible(False)
fig.savefig('research/elementary-probe.png',dpi=160);plt.close(fig)
rows='\n'.join(f'| {label} | {mean(r["retina_plastic"][key]):.2f}% | {mean(r["rewired_plastic"][key]):.2f}% |' for key,label in zip(keys,labels))
text=f'''# Diagnóstico de una pista: dónde resulta accesible la información

13 de septiembre de 2026. Corrida independiente terminada: `runs/elementary-probe-001`.

## Resultado

Leyendo la vecindad tras tres ciclos, una pequeña red diagnóstica clasifica correctamente {mean(r['retina_plastic']['cycle3']):.2f}% de las posiciones. Leyendo solo la actividad de la casilla, {mean(r['retina_plastic']['cycle3_target']):.2f}%. Con la vecindad comprimida a diez valores mediante PCA ajustado exclusivamente en entrenamiento, {mean(r['retina_plastic']['neighborhood_pca10']):.2f}%. Este último control usa el mismo ancho de entrada y la misma arquitectura del lector que la casilla aislada.

Esto **apoya revisar el acceso espacial del decoder** antes de sustituir toda la dinámica neuronal. No establece pérdida irreversible de información: es información accesible a un lector concreto. No implica que el modelo haya aprendido a jugar mejor: el conectoma está congelado, se entrenaron lectores diagnósticos nuevos.

![Gráfica](elementary-probe.png)

## Tarea y separación

Tableros 5×5, tres minas, pistas correctas calculadas a partir de posiciones de minas. Se designa una pista interior con dos o tres vecinos ocultos. Clases equilibradas: todos seguros desde esa pista (pista cero); todos minas (pista igual a número de vecinos ocultos); indeterminados **desde esa pista** (pista uno y varios vecinos ocultos). Otras pistas pueden resolver una situación etiquetada aquí como indeterminada. El lector recibe la ubicación de la pista designada mediante la extracción de su vecindad.

Estas son aperturas parciales controladas: **se desactiva conceptualmente la expansión automática de ceros**, por lo que no todas estas observaciones son alcanzables en el entorno actual. Esta tarea mide una regla, no win rate ni detección general de minas. No se usan banderas ni posiciones de minas como entrada.

600 posiciones de entrenamiento, 240 de validación, 240 de prueba; 80 por clase en prueba. Se separan las 84 máscaras posibles de vecinos ocultos en 50/17/17 máscaras antes de generar observaciones. No hay máscaras ni tableros visibles compartidos entre particiones. Las rotaciones/reflejos no se agrupan: no afirmamos generalización a familias de simetría nunca vistas. Las 240 posiciones comparten 17 máscaras reservadas y no deben tratarse como 240 familias independientes.

## Modelos y lectores

Se cargaron los checkpoints de 3,000 actualizaciones de `retina_plastic` y `rewired_plastic`, semilla 20260926. Solo inferencia, sin modificar sus pesos. Extracción en batches de 16 sobre los grafos completos. El score extraído coincide con el forward original: error máximo {v['retina_plastic']['forward_max_error']:.3g} y {v['rewired_plastic']['forward_max_error']:.3g} respectivamente.

Lectores MLP con 64 unidades tanh y tres salidas; AdamW, lr 0.003, weight decay 0.01, máximo 600 updates completos. Estandarización ajustada solo en entrenamiento. Selección cada 20 updates por pérdida de validación, nunca por prueba. Tres semillas del lector (0/1/2), **no tres entrenamientos nuevos del conectoma**. La anchura varía según la representación; el control PCA10 permite una comparación de igual anchura con los diez tipos neuronales de una casilla. PCA usa solo entrenamiento y no etiquetas.

| Información disponible para el lector nuevo | Original | Reconfigurado |
|---|---:|---:|
{rows}

CNN pequeña entrenada desde cero para esta tarea, con el mismo parche visible y las mismas particiones: {', '.join(str(x['test_correct'])+'/'+str(x['n']) for x in cnn)}. Confirma que el diagnóstico es aprendible por una arquitectura convencional bajo esta separación; no es el checkpoint CNN de partidas completas.

## Interpretación y límites

- La información para esta regla resulta legible tras la propagación: no observamos un derrumbe progresivo entre los ciclos 1 y 3.
- El paso de vecindad a una única casilla reduce la decodificación. La ventaja de PCA10 frente a casilla10 sugiere que importa **qué contexto** recibe el lector, no solo cuántos valores recibe.
- El decoder actual fue entrenado para puntuar acciones, no para clasificar tres estados de una pista. Su score escalar no necesita preservar toda esta clasificación; su baja precisión no es por sí sola una prueba de un bug.
- Los parches se centran en la pista designada; el decoder real puntúa una casilla sin que se le designe una pista. Transferir la mejora a juego completo requiere una intervención y nueva evaluación.
- Ambos grafos conservan información útil; esto no establece superioridad anatómica del conectoma original. Son checkpoints de una sola semilla.
- Una prueba positiva de información accesible tampoco demuestra razonamiento causal ni que el entrenamiento anterior utilizara esa regla.

## Siguiente intervención propuesta

Un lector espacial sobre el mapa de actividad neuronal: para puntuar cada casilla, combinar actividad de su entorno, por ejemplo dos convoluciones 3×3 (campo 5×5). Ese campo puede incluir una pista vecina y todos sus vecinos. No darle directamente las pistas al decoder: mantener la decisión dependiente de actividad del conectoma.

Congelar inicialmente encoder y cerebro, entrenar solo el lector sobre posiciones reales ya disponibles, comparar con el lector actual y evaluar partidas nuevas sin maestro por dificultad. No se lanzó esa intervención en este diagnóstico.

## Reproducibilidad

Scripts: `experiments.elementary_probe`, `experiments.elementary_cnn`, `experiments.elementary_target_probe`, `experiments.verify_elementary`, `experiments.report_elementary`, en ese orden, con `.venv/bin/python -m`. El primer script rechaza sobrescribir corrida completada. Los demás regeneran artefactos derivados. Datos, features, resultados por semilla, verificación y copia de fuentes en la carpeta de corrida. El checkpoint híbrido conserva SHA256 `{v['protected_hybrid_sha256']}`.
'''
Path('research/RESULTADO_HABILIDAD_ELEMENTAL.md').write_text(text)
source=OUT/'source';source.mkdir(exist_ok=True)
for name in ['elementary_probe.py','elementary_cnn.py','elementary_target_probe.py','verify_elementary.py','report_elementary.py','retina_policy.py','plastic_sparse.py','plastic_kernel.cpp']:
 shutil.copy2(Path('experiments')/name,source/name)
(OUT/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2))
print('Report and chart saved')
