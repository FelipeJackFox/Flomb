# Selección de checkpoints mediante partidas

Experimento sin reentrenamiento: tres semillas y dos brazos (uniforme/prioritario). Para cada brazo se consideran exclusivamente tres candidatos ya guardados: DAgger anterior, mejor por pérdida y último de la corrida de priorización. No están disponibles todos los checkpoints intermedios, por lo que esta prueba no mide búsqueda exhaustiva.

Se reservan previamente 500 layouts7×7/7 minas de validación y otros500 finales, nuevos y separados de todos los datasets/benchmarks/colección anteriores. Registro también en dataset.pkl para excluir ambos conjuntos de futuras reservas. Apertura central segura idéntica; excluir victorias automáticas de las métricas.

Selección por juego: mayor número de victorias autónomas en validación; desempate por menos actualizaciones adicionales y después orden fijo (anterior/mejor/último). Comparador: checkpoint ya elegido por pérdida. Selecciones de los seis brazos escritas y selladas por SHA256 antes de cualquier evaluación final. Solo se evalúa en el conjunto final la unión de elegidos por juego/pérdida y baselines; no se evalúan allí los candidatos descartados para retocar selección.

Deduplicar evaluaciones solo cuando todos los pesos del head son idénticos; mismo cerebro/encoder/mapa congelados. Partidas y actividad pueden compartirse por observación/contexto, nunca decisiones entre heads diferentes. Sin entrenamiento ni actualizaciones de Adam, sin maestro ni máscara de minas durante inferencia.

Comparar métodos por brazo y semilla sobre los mismos tableros finales. Bootstrap emparejado por tablero de la diferencia media entre lectores, condicionado a ellos; no tratar repeticiones de layouts como independientes. Verificar sellado, splits, consistencia de reglas, hashes y resultados por partida. No sustituir agente servido.
