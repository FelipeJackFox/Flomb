"""Evidence and concrete next intervention, without modifying the policy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiments.diagnose_mine_choices import OUT
from experiments.expand_dagger_experience import SEEDS,target
from experiments.capacity_probe import write_json

def main():
 assert (OUT/'completed.json').exists()
 replay=json.loads((OUT/'replay.json').read_text());events=json.loads((OUT/'mine-events.json').read_text());cached=json.loads((OUT/'cached-summary.json').read_text());fit={}
 for seed,sets in cached.items():
  used=[r for r in sets.values() if r['available_to_selected_model']]
  fit[seed]={k:sum(r.get(k,0) for r in used) for k in ('positions','safe_choices','known_mine_choices','single_clue')}
 summary=dict(replay=replay,available_training_positions=fit,tiny_overfit=json.loads((OUT/'tiny-overfit.json').read_text()),scope='replay one seed; cached fit three seeds; tiny in-sample fit does not test generalization')
 write_json(OUT/'summary.json',summary)
 example=next(e for e in events if e['proof']=='single_clue' and e['has_safe']);write_json(OUT/'illustrated-example.json',example)
 fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained');keys=['single_clue','local_subset','global_count','exact_enumeration'];labels=['Una pista','Combinar pistas','Conteo global','Enumeración exacta'];v=[replay.get(k,0) for k in keys]
 ax[0].barh(labels,v,color='#078493');ax[0].invert_yaxis();ax[0].set_xlabel('Clics en minas deducibles');ax[0].set_title('500 partidas · lector 20261002')
 for i,n in enumerate(v):ax[0].text(n+.5,i,str(n),va='center')
 ax[0].set_xlim(0,max(v)*1.2+1)
 b=np.array(example['board']);chosen=example['action'];clue=example['witness'][0]['clue'];ax[1].set_xlim(0,7);ax[1].set_ylim(7,0);ax[1].set_aspect('equal');ax[1].axis('off');ax[1].set_title(f"Ejemplo real · partida {example['game_seed']}")
 for r in range(7):
  for c in range(7):
   i=r*7+c;color='#dee5e9' if b[r,c]<0 else '#f6f6f6'
   if i==chosen:color='#f08c86'
   if i==clue:color='#f8cb70'
   ax[1].add_patch(Rectangle((c,r),1,1,facecolor=color,edgecolor='white'))
   text='×' if i==chosen else ('?' if b[r,c]<0 else str(b[r,c]) if b[r,c] else '')
   ax[1].text(c+.5,r+.5,text,ha='center',va='center',fontsize=17)
 ax[1].text(3.5,7.5,'Rojo: clic del lector · Amarillo: pista suficiente',ha='center',fontsize=9)
 fig.savefig('research/mine-choice-diagnostic.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {s} | {r['positions']} | {r['known_mine_choices']} ({100*r['known_mine_choices']/r['positions']:.2f}%) | {r['single_clue']} |" for s,r in fit.items())
 doc=f'''# Diagnóstico de clics en minas deducibles

Se auditó sin cambiar los modelos originales: ajuste sobre posiciones de entrenamiento disponibles para los tres lectores ampliados, validación original y repetición exacta de 500 partidas del lector 20261002. La repetición coincide partida por partida con victorias, clics, minas conocidas elegidas y oportunidades/elecciones seguras registradas anteriormente.

## Qué ocurre en partidas reales

De {replay['known_mine_choices']} clics en minas deducibles, **{replay.get('known_mine_without_safe',0)} ocurrieron sin ninguna jugada certificada segura** y **{replay.get('known_mine_with_safe',0)} teniendo al menos una alternativa segura**. El primer grupo no recibe supervisión en el esquema DAgger actual; el segundo sí pertenece al tipo de estado que entrenamos. En esta repetición el primer grupo fue cero: omitir estados sin solución segura NO explica los 176 errores observados. En todos había alguna casilla no certificada como mina. Elegir una mina demostrable no era una apuesta 50/50 inevitable.

Complejidad suficiente de prueba (clasificación jerárquica, no única): una pista {replay.get('single_clue',0)}, combinación de pistas locales {replay.get('local_subset',0)}, conteo global {replay.get('global_count',0)}, enumeración exacta {replay.get('exact_enumeration',0)}. La clasificación local usa el mismo cierre acotado del solver, por lo que no identifica necesariamente la prueba matemática más corta posible.

![Diagnóstico](mine-choice-diagnostic.png)

El ejemplo muestra únicamente la observación pública. La pista amarilla tiene tantos vecinos ocultos como minas indicadas, de modo que todos esos vecinos son minas. El lector escogió uno de ellos (rojo). No se usó el mapa oculto para certificarlo. El ejemplo también tenía una jugada segura disponible según el solver.

## El fallo también aparece en datos disponibles durante entrenamiento

| Lector | Posiciones disponibles | Elige mina deducible | De ellas, bastaba una pista |
|---|---|---|---|\n{table}

Son evaluaciones del checkpoint final sobre el conjunto que tenía disponible, no partidas independientes ni prueba de que cada fila apareciera en un minibatch. Se excluyen rondas posteriores al checkpoint seleccionado y se conserva la experiencia heredada. Para todos estos estados existía alguna acción segura etiquetada. La dificultad no se explica solamente por encontrar tableros nuevos o por omitir estados de apuestas.

## Qué enseña el objetivo actual

`equivalent_loss` pone una distribución uniforme sobre todas las jugadas certificadas seguras. Las demás casillas tienen objetivo cero: una mina conocida y una casilla de riesgo incierto no reciben etiquetas de riesgo diferentes. Sí hay penalización indirecta a la mina por softmax; sería incorrecto decir que se ignora por completo. Una prueba verifica que intercambiar logits entre dos acciones no seguras deja la pérdida igual.

Además, el colector descarta estados sin jugada segura. Por tanto, no enseña en esos estados a descartar minas demostrables y elegir entre las apuestas restantes. El enmascaramiento de inferencia solo excluye casillas ya abiertas/fuera del tablero; una mina deducible sigue siendo una acción legal. El replay exacto no mostró discrepancias de ejecución respecto a la evaluación anterior.

## Qué podemos concluir y qué falta

Hay errores residuales de ajuste incluso sobre posiciones disponibles. El hueco de estados sin solución segura existe, pero no explica los clics en minas observados en estas 500 partidas. La selección de acciones en el replay coincidió exactamente con la evaluación original; no apareció una discrepancia de índices o ejecución.

Una prueba adicional entrenó una copia temporal del lector sobre 32 errores heredados que se resolvían con una sola pista. Con la misma actividad congelada, arquitectura y pérdida, Adam nuevo lr 0.001, batch fijo de 32 y clipping 5, pasó de 0/32 a 32/32 decisiones seguras en 50 actualizaciones y mantuvo ese resultado hasta 500. No se guardó ni sirvió esa copia como nuevo agente. Es ajuste a un pequeño conjunto seleccionado por sus fallos, no prueba de generalización ni de aprendizaje de una regla abstracta. Sí descarta que estos 32 fallos particulares sean imposibles de corregir con la interfaz actual.

No se separan causalmente representación, capacidad y optimización para el resto del juego, ni se demuestra que la dinámica simplificada sea la causa. La pérdida actual también pudo ajustar esos 32 casos, por lo que todavía no hay evidencia de que sustituirla sea necesario.

**Siguiente experimento propuesto:** mantener cerebro, encoder, arquitectura y pérdida, y priorizar dentro del replay las posiciones de entrenamiento donde el lector elige una mina certificada pese a tener una alternativa segura. Mezclarlas con experiencia ordinaria para evitar olvidar otras situaciones. Comparar con muestreo uniforme a igual presupuesto y evaluar reducción de errores más victorias en tableros finales nuevos. Así se aísla primero la distribución de entrenamiento. Una tarea auxiliar por casilla para distinguir segura/mina podría evaluarse después, por separado. La priorización no se implementó ni entrenó en esta fase; solo se hizo la prueba temporal de ajuste a 32 ejemplos.

El benchmark previamente final se inspeccionó ahora con fines de diagnóstico: cualquier intervención posterior debe usar una evaluación final nueva. Fuentes, ejemplos completos, contadores y hashes en runs/mine-choice-diagnostic-001. Dos pruebas diagnósticas pasan; checkpoints originales intactos. El análisis de trayectoria cubre una semilla y los resultados de ajuste tres lectores sobre un único cerebro.
'''
 Path('research/DIAGNOSTICO_MINAS_DEDUCIBLES.md').write_text(doc);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
