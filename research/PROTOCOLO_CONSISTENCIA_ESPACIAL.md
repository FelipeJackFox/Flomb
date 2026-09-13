# Consistencia del decoder espacial

Corrida autorizada: spatial-consistency-001. Objetivo: comprobar si la ventaja del lector espacial se mantiene al cambiar semilla del lector, sin ajustar arquitectura/presupuesto por resultados finales.

- Semillas 20261002 (checkpoint del piloto, reutilizado sin cambio), 20261003 y 20261004 (nuevas).
- Comparadores: original congelado, tres lectores puntuales y tres espaciales de capacidad comparable (~12.8k parámetros).
- Los dos nuevos pares repiten 1,000 updates, batch64, Adam lr0.001, clip5, minibatches balanceados por tamaño/categoría. Selección por validación cada100 updates.
- Se reutilizan los mismos 10,000 estados de entrenamiento y 1,000 de validación, sus cachés y el mismo checkpoint cerebral. Esta prueba varía el lector, no la semilla del cerebro ni el split de entrenamiento.
- 500 partidas 7×7/7 minas reservadas antes del entrenamiento, excluyendo layouts de datasets y benchmarks previos. Todos los modelos juegan los mismos tableros, sin maestro, argmax legal. No usar 5×5 como evaluación inédita: su universo ya se agotó.
- La memoización de actividad depende solo de observación pública y contexto; se verifica contra la inferencia original y al reordenar un batch mixto. No se guardan ni reutilizan acciones del maestro.
- Reportar cada semilla, comparación emparejada por tablero y promedio entre lectores. Tres semillas en las mismas 500 partidas no son 1,500 tableros independientes. Intervalos por remuestreo de tableros, si se calculan, quedan condicionados a estos tres lectores.
- Conservar checkpoints originales y no desplegar/adoptar automáticamente el ganador.
