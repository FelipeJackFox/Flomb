# Repetir adaptación a9×9 en otras dos políticas

El piloto02 mejoró en9×9 (13/500 vs2/500control y4/500baseline), pero bajó en7×7 (42/250 vs45control y53baseline). Antes de modificar el método se repite exactamente su presupuesto, regla de último checkpoint2250 y mezcla de datos en las políticas03/04 preservadas.

Dos nuevos entrenamientos por pares:3×300partidas9×9/12 autónomas y3×750updates por brazo, heredando pesos/Adam/RNG de cada `wide-reader-001/best-SEED-local.pt`. Control64originales5/7,DAgger32originales+32experiencia9acumulada. Grafo/encoder congelados, mapeo extendido idéntico, docente solo etiqueta. Las dos nuevas políticas comparten900layouts de colección pero generan trayectorias propias.

Test final500layouts9×9 y2507×7 NUEVOS, compartidos por las tres políticas (piloto02 se reevalúa sin entrenarlo). Reservar900colección+750test disjuntos de históricos antes de iniciar; sellar todos los candidatos finales antes de evaluar. Último2250 fijo en ambos brazos, mejores de pérdida antigua solo diagnóstico.

Principal: diferencia media DAgger−control en9×9 de las DOS semillas nuevas; secundaria: media de tres, comparación con baseline y retención7×7. Mostrar cada semilla, conteos y oportunidades seguras. Bootstrap por layout compartido entre semillas:750layouts, no2250 independientes ni tres cerebros. No decidir ganador con datos de colección ni ocultar regresiones en7. Conservar agente servido y checkpoints originales.

Coordinador `runs/nine-dagger-consistency-001`; hijos `runs/nine-dagger-20261003/4`. Runner `experiments/repeat_nine_dagger.py`, cierre `experiments/report_nine_dagger_consistency.py`.
