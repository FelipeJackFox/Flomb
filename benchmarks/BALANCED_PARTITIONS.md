# Reparto por conexiones · 12 septiembre 2026

Diagnóstico del híbrido: últimas100partidas~575.5s (5.75s/partida). A ritmo constante,50k serían~80h; no es una ETA fiable porque cambia la longitud y dificultad. Todavía no había evaluación periódica. Métricas observacionales ~milisegundos o menos; cProfile de una actualización:2.001s, de los cuales1.794s en multiplicaciones cerebrales dispersas.

Cuatro bloques iguales en neuronas tenían11,184,130 /4,844,834 /3,042,331 /6,511,643 conexiones en forward. Se cambian límites de filas para aproximar un cuarto de conexiones por hilo; filas completas, acumulación original y toda la red conservadas. Backward usa el mismo criterio sobre la transpuesta.

Prueba emparejada e intercalada, batch16,4workers, red completa, tres repeticiones por variante, junto a las corridas activas:

- Mediana original forward+backward:0.99894s.
- Mediana equilibrada:0.66636s.
- Cociente:1.4991×. Features y gradientes exactamente iguales en todas las repeticiones.

Este factor mide el núcleo forward+backward, no rendimiento total del entrenamiento ni calidad. No se multiplican speedups históricos. No se reducen actualizaciones, neuronas, pasos temporales ni batch. No se congelan ganancias cerebrales. La corrida anterior permanece intacta.

Se reanuda hybrid-001 desde un checkpoint completo nuevo; snapshot previo y metadatos en pre-balanced-EPISODIO y balanced-migration.json. Se conserva optimizer, replay, redes online/target y RNG. Cualquier observación posterior al checkpoint se conserva en el snapshot de migración, fuera de las curvas activas.
