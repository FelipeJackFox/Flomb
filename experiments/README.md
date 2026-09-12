# Tres entrenadores independientes

- `dagger`: etiquetas del solver visible en todos los estados visitados; mezcla maestro/alumno decreciente; clasificación sobre acciones equivalentes. Dataset agregado con muestreo uniforme.
- `qrdqn`: cabeza dueling, Double DQN, 16 cuantiles configurables, replay priorizado con importancia y retornos de 3 pasos. Sin maestro.
- `hybrid`: mismo QR-DQN + correcciones DAgger y pérdida de imitación separada, de peso decreciente.

Es una variante QR-DQN con componentes de Rainbow, **no Rainbow completo**: no incluye NoisyNet, y usa cuantiles en lugar de C51. No se copió ni se reutilizó el checkpoint de Street Fighter.

Las tres entrenan las ganancias internas del conectoma completo y su propia cabeza. Comparten el mismo checkpoint inicial `runs/curriculum-001/initial-expanded.npz` (con el antecedente del piloto anterior) y semilla de inicialización. Conservan topología, tres ciclos y proyección anatómica/artificial del modelo existente. Se añaden a la cabeza el tamaño y densidad total de minas, datos públicos del tablero; nunca ubicaciones ocultas. Estos datos también los conoce el solver.

## Ejecutar un piloto comparable

Desde la raíz del proyecto:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.run_suite --root runs/method-pilot-002 --episodes 64 --eval-per-stratum 8
```

Ejecuta los tres secuencialmente para evitar contención. Ver `suite-state.json`, logs por método, checkpoints y `comparison.json`. Para recuperar esa misma suite tras una interrupción, repetir el comando añadiendo `--resume`. Reanuda el método interrumpido y omite los terminados. No toca `curriculum-001`.

## Corridas largas por bloques

Ejemplo de preparación de una corrida nueva de 50,000 episodios, **no lanzada automáticamente**:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.train --mode hybrid --run runs/hybrid-long-001 --episodes 50000 --stop-after 128 --eval-per-stratum 32
```

Repetir exactamente la configuración con `--resume --stop-after 128` para otros 128 episodios. `--stop-after` limita la invocación; no cambia los calendarios ni declara entrenamiento completo. El objetivo, batch y demás parámetros permanecen inmutables al reanudar. Cambiar algoritmo/configuración requiere otra carpeta. Checkpoint inicial y cada 16 episodios (también al final de cada bloque), escritura atómica. Un corte imprevisto recupera la última frontera guardada; no el episodio incompleto. El pickle contiene solamente estado local confiable: no cargar checkpoints externos no confiables.

La ayuda del maestro llega a cero al 60% del presupuesto total `--episodes`; epsilon llega a .05 al 50%. El currículo de tamaños mantiene por defecto un horizonte independiente de 50,000: un piloto corto prueba mecánica de aprendizaje, no reproduce las 50,000 partidas de dificultad creciente. Para un currículo corto completo se puede definir `--curriculum-horizon` al crear otra corrida. El híbrido conserva peso de imitación mínimo .1 con datos ya agregados, aunque la ayuda durante acciones llegue a cero.

## Optimizaciones implementadas y límites

- SpMM CSR para batch; batch16 por defecto según mediciones, no matrices densas NxN.
- Grafo y transpuesta compartidos entre online/target; target separa ganancias entrenadas.
- Observaciones int8: 256 bytes por estado frente a 10,240 bytes en one-hot float32. Las etiquetas booleanas usan 32 bytes. Se materializa one-hot solo para el batch.
- Se conservan las activaciones solo durante una actualización. Nunca se reutilizan características antiguas tras cambiar ganancias internas.
- No se procesa cerebro target en terminales ni se calcula QR Huber para acciones no ejecutadas.
- Durante una acción forzada por maestro se omite la inferencia del alumno; las etiquetas/experiencias siguen entrando al entrenamiento. Esto es teacher forcing explícito, no una acción supuestamente autónoma.
- Un hilo BLAS por proceso; no render, video, ni pausas de tiempo real durante entrenamiento.
- Por defecto una actualización batch16 cada cuatro clics, limitada por warmup. Este ratio es una elección de entrenamiento, no una aceleración matemáticamente equivalente a otro ratio.
- Kernels CPU NumPy/SciPy; no se ha medido una implementación CUDA/MPS. Más actores en esta Mac no implica mayor velocidad.

Mediciones y fuentes: `research/OPTIMIZACIONES_ENTRENAMIENTO.md`, `benchmarks/batch_probe_result.json`, `benchmarks/backbone_batch_probe_result.json`. El 1.41x observado es forward+backward del backbone B16 frente a serial, no velocidad global frente al entrenador antiguo de cuatro hilos.

## Evaluación e integridad

Evaluación greedy, sin solver, sin epsilon ni actualización, con seis estratos y semillas desde 2,000,000,000 (las de la evaluación final original). Las aperturas automáticas se informan aparte. La política original se evalúa estocásticamente en su programa histórico: hay que reevaluarla greedy antes de una comparación homogénea. Las nuevas cabezas y entradas públicas también difieren de esa referencia; no atribuir toda diferencia solo al algoritmo.

Los pilotos de 64 episodios/8 tableros por estrato son diagnóstico de ejecución y señales tempranas. No prueban superioridad, razonamiento ni ventaja del conectoma. Registrar partidas, clics, updates, segundos y memoria; comparar calidad por cómputo con varias semillas en la siguiente etapa.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest experiments.test_backbone experiments.test_qr_core
```

Pruebas: gradientes numéricos de backbone batched, cabeza dueling y QR-Huber; padding y máscara; maestro independiente de minas ocultas; targets terminales; replay/Adam reanudados. Prueba real de DAgger por bloques 2+2 frente a 4 continuos coincidió exactamente en parámetros, Adam, target, replay y RNG (versión smoke anterior a añadir los dos datos públicos de contexto). Registros históricos conservados en `runs/method-smoke-*`.

## Cuatro hilos, verificados después del primer piloto

El backend actual utiliza `--workers 4` por defecto para repartir filas de cada
multiplicación dispersa; `--workers 1` conserva la ejecución serial. Las particiones
comparten datos/índices del CSR. En B16 se midieron 1.59 s con un hilo, 1.11 s con
dos y 0.88 s con cuatro, forward+backward del cerebro; resultados y gradientes
idénticos. No es una medición del proceso completo ni debe multiplicarse ciegamente
por otros speedups. El primer conjunto `method-pilot-001` se lanzó antes de activar
esta optimización y utiliza un hilo disperso; sus tiempos no representan el backend nuevo.

`--workers` se guarda como cómputo por invocación, no altera el calendario del
algoritmo. Se comprobó un híbrido actual (incluyendo los dos inputs públicos):
dos episodios con cuatro hilos y reanudación de otros dos con uno, frente a cuatro
continuos con un hilo, coinciden exactamente en pesos, Adam, target, replay, RNG,
contadores y pérdidas. Evidencia: `benchmarks/resume_equivalence.json`.
Cada invocación nueva guarda código y hashes en `invocations/` y `compute-*.json`.
