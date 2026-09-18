# ¿Aprende el cerebro? Plasticidad sináptica con interfaz congelada (prefijado 18 sept 2026, antes de lanzar)

Motivo: toda la línea reciente entrenó encoder y lector alrededor de un conectoma congelado, y con 1 ciclo de propagación.
Conteo sobre el grafo: con 1 ciclo solo 23,778 de 25,582,938 conexiones están en el camino entrada→salida (un salto L1‑L3 → Mi/Tm);
el resto del cerebro no participa. Esta prueba invierte el reparto: la interfaz queda fija y lo único que aprende son las sinapsis.

Configuración: grafo MaleCNS y signos fijos, 3 ciclos, punto de partida retina_plastic-20260926, encoder early-interface-001/latest-joint
congelado en todo el experimento. Seed 20261111, batch 64 (16 por cada una de las 4 fuentes históricas, microbatch 16), Adam .001, clip 5.

1. Calentamiento común: lector nuevo ActivityHead (12769 parámetros), 3000 updates sobre la actividad de 3 ciclos del cerebro intacto → `baseline`.
2. `reader` (control): el lector sigue 3000 updates más; cerebro fijo.
3. `brain`: encoder y lector congelados; 3000 updates sobre 25,582,938 log-ganancias por conexión y 166,700 por neurona, con exactamente
   las mismas posiciones que `reader` (mismo RNG continuado; se verifica por bloques de 250).

Evaluación: una inferencia por jugada, 750 layouts nuevos compartidos (500 de 9×9/12 minas, 250 de 7×7/7 minas). Principal: 9×9 brain − baseline.
Secundarios: brain − reader, reader − baseline, y 7×7. Bootstrap pareado 10000. Endpoint fijo, sin selección por test, sellado antes de evaluar.
Se reporta cuántas conexiones cambiaron y cuánto. Validado antes de lanzar (experiments/test_synaptic_learning.py): paridad con la ruta de
inferencia, gradiente solo hacia sinapsis, derivadas contra diferencias finitas en float64, y que las ganancias aprendidas cambian la inferencia.
Preflight real en el arranque: paridad de forward y gradiente no nulo en conexiones.

Límites: modelo de tasas tanh, no spikes ni EPSPs. Una semilla, un cerebro. Un resultado favorable NO demuestra ventaja anatómica: eso exige
repetir con el grafo recableado (runs/structural-learning-001/rewired.npz) como control, que es el paso siguiente si hay señal.
Presupuesto estimado 3–5 h de CPU local. Agente servido, checkpoints históricos y originales intactos.
