# Comparación controlada de entradas

Autorizada el 13 septiembre 2026. `runs/input-representation-001`.

Tres pares de semillas 20261002/3/4. En cada par: lector espacial idéntico de 12,769 parámetros (campo receptivo 5×5), inicialización y secuencia de minibatches idénticas, Adam lr0.001, clip5, batch64 y 5,000 actualizaciones completas. Diez canales de pistas públicas one-hot frente a diez canales de actividad del cerebro congelado; ambos reciben tamaño y número público de minas. Pérdida actual de imitación de todas las acciones certificadas seguras. Ningún filtro del maestro en la elección autónoma.

10,000 posiciones originales y 1,000 holdout, mismas filas, etiquetas y máscaras. Sin nueva experiencia DAgger. Ambos lectores comienzan desde cero; el cerebro/encoder ya fue entrenado previamente y queda congelado. Igual presupuesto adicional del lector no significa igual costo histórico total. Escalas de entrada originales, sin ajustar según resultados.

Selección por pérdida mínima de holdout en paso0 y cada250, empate más temprano. Se sella antes de evaluar 500 layouts nuevos compartidos de 7×7/7 minas, excluidos de datasets y listas previas de juegos. No se usa el resultado final para escoger checkpoints. Excluir victorias automáticas de apertura del denominador. Comparación pareada por tablero con tres semillas de lector; no son tres cerebros independientes ni 1,500 tableros independientes.

Comprobar igualdad de metadatos de caché, recomputar 16 muestras contra el cerebro, igualdad de inicialización y minibatches por hash, integridad de fuentes/checkpoints protegidos. Conservar pesos mejores, últimos, Adam y RNG. No sustituir agente servido.

Interpretación: diferencia a favor de pistas directas indicaría un coste de la representación cerebral actual para este lector/presupuesto. No separa codificación, dinámica y pooling, ni demuestra incapacidad del conectoma o de otras interfaces. Si ambos fallan, investigar lector/objetivo/datos. El holdout es histórico y reutilizado: la prueba final nueva es la evidencia de generalización. No comparar directamente estos resultados con el ~20% de modelos refinados con DAgger en otros benchmarks.
