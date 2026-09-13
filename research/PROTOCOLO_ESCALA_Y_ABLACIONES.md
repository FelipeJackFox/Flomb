# Más demostraciones y separación de componentes

Se ejecuta el siguiente paso aprobado: ampliar datos, comprobar aprendizaje autónomo con un control barato y comparar componentes del conectoma. El híbrido en episodio 14,384 sigue protegido.

## Datos

10,000 posiciones distintas: 5,000 de 5×5/3 minas y 5,000 de 7×7/7 minas. Pueden provenir varias posiciones del mismo tablero, pero todas pertenecen al mismo split. Entrenamiento tiene 1,555 layouts; validación tiene 1,000 posiciones de 146 layouts. Se reservan primero 500 partidas finales, 250 por tamaño, sin excluir aperturas difíciles. Estas partidas tampoco aparecen en los datasets de los pilotos espacial y de capacidad anteriores.

El maestro solo usa información visible. Los hashes del mapa oculto se usan exclusivamente para separar layouts. Validación y entrenamiento no comparten estados visibles. Entrenamiento contiene 1,066 posiciones relacionales/exactas 5×5 y 336 de 7×7; validación contiene 140 y 39. Los minibatches equilibran los cuatro grupos tamaño/tipo, por lo que los grupos pequeños se repiten más.

## Presupuestos decididos antes de las ablaciones

- Control CNN inicial: 3,000 actualizaciones, batch 64, semillas 20260924 y 20260925. Sirve para comprobar si esta receta aprende partidas completas con mayor experiencia.
- Comparación: 3,000 actualizaciones, batch 16, semilla 20260924. CNN, conectoma actual, solo encoder aprendido, solo dinámica interna nueva y ambos. Es un primer análisis por componentes con una semilla; una ventaja requerirá réplica antes de adopción.
- Todos los modelos de la comparación reciben los mismos minibatches, Adam a 0.001 y pérdida uniforme sobre todas las acciones seguras etiquetadas. Las cuatro variantes del conectoma usan un canal y la misma cabeza inicial. No son la variante anterior de dos canales ni una copia del entrenamiento QR-DQN.

La dinámica interna nueva agrupa sesgos, retención, reinyección y transformación compartida. La separación identifica el efecto del encoder frente a ese conjunto; no identifica cada mecanismo interno. Toda la conectividad se conserva y las ganancias internas siguen aprendiendo.

## Ejecución y recuperación

Scripts: `experiments/scaled_data.py`, `experiments/scaled_train.py`, `experiments/report_scaled_training.py`. Guardado atómico cada 100 actualizaciones con pesos, Adam, RNG y contador. `--resume` exige el mismo manifiesto, código, dataset y checkpoint protegido; no sobrescribe resultados terminados. Se pasó una prueba real del CLI interrumpido/restaurado frente a ejecución continua, con pesos y estados Adam exactamente iguales.

Las evaluaciones de partidas se agrupan por lotes para amortizar la propagación del grafo. La evaluación agrupada coincide con la serial en la prueba. Se registran victorias, aperturas automáticas, oportunidades seguras y elecciones seguras, muertes cuando existía una acción segura y casos de riesgo inmediato mínimo exacto del 50%. El último contador no altera las victorias reales.

El presupuesto de la comparación es cuatro veces menor en ejemplos procesados que el de la CNN inicial. Por eso incluye otra CNN con presupuesto idéntico al conectoma. No se seleccionan checkpoints usando las partidas finales. Solo se evalúa el estado inicial y el final en las posiciones reservadas.

Los resultados viven en `runs/scaled-learning-001`. El entrenamiento largo anterior no se reanuda. Esta comparación finita se supervisa hasta finalización o fallo, avisando solo cuando haya un resultado o una acción necesaria.
