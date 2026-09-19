# Entrada legible (prefijado 18 sept 2026, antes de lanzar)

Motivo: la sonda information-probe-001 mostró que el tablero entra 98% recuperable y tras el primer salto sináptico solo 83% de los
números es legible (exactitud balanceada 0.94 → 0.61), sin empeorar con más ciclos. Ese salto es el cuello de botella.

Etapa 1 (experiments/train_legible_entry.py), 1 ciclo, seed 20261113, 3000 updates, batch 64 (16×4 fuentes), mismos draws en ambos brazos.
Objetivo único: reconstruir la pista de cada casilla (tapada, 0..8) desde la actividad de salida con un decodificador auxiliar 3×3.
- encoder: aprende el encoder (lr .0003) y el decodificador auxiliar (.001); conectoma fijo.
- synapses: además aprenden las log-ganancias por conexión y por neurona (.001); topología y signos fijos.
Las 150 partidas reservadas de la sonda quedan excluidas del entrenamiento por layout_hash en todas las fuentes.

Etapa 2 (experiments/evaluate_legible_entry.py), tres entradas: baseline (original), encoder, synapses.
1. Sonda independiente: decodificadores NUEVOS (lineal 1×1, lineal 3×3, MLP 3×3) entrenados en 3000 posiciones y medidos en 1000 de las
   150 partidas reservadas. El decodificador auxiliar de la etapa 1 no se usa para medir.
2. Lector de juego nuevo ActivityHead (12769 parámetros), misma inicialización y draws, 3000 updates sobre la actividad de cada entrada.
3. 750 layouts nuevos compartidos (500 de 9×9, 250 de 7×7), una inferencia por jugada.
Principal: 9×9 synapses − baseline. Secundarios: encoder − baseline, synapses − encoder, 7×7, y acierto de la sonda. Bootstrap pareado 10000.
Endpoint fijo, entradas selladas por hash antes de evaluar. Una semilla: resultado favorable exige réplica. Agente servido/originales intactos.
