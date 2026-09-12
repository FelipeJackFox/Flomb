# Referencia lógica de Buscaminas

Este módulo es un solucionador de referencia independiente: **no es la política neuronal de la mosca**, no modifica pesos y no participa en el entrenamiento.

## Ejecutar

Desde la raíz del proyecto:

```sh
.venv/bin/python -m unittest test_solver -v
nice -n 15 .venv/bin/python evaluate_solver.py --episodes 192
```

Resultados completos por episodio: `reference/results/benchmark.json`. El benchmark compara aleatorio y solver sobre las mismas semillas y la misma apertura automática central segura del entorno. Usa seis configuraciones iguales a la evaluación del currículo: 5×5/3 minas, 7×7/7, 9×9/10, 12×12/24, 16×16/40 y 16×16/56. Las semillas empiezan en 3,000,000,000, fuera de los rangos actuales de entrenamiento, validación y evaluación final. No se evalúa aquí la red neuronal.

## Qué sabe y cómo decide

`solver.choose(observation, legal_mask, size, mine_count, rng)` recibe exclusivamente la observación one-hot **sin padding**, máscara legal, tamaño y minas totales públicas. No acepta el objeto entorno ni su estado oculto. `solver.analyze(...)` permite clasificar una acción de cualquier política como `proven_safe`, `proven_mine` o `guess`.

1. Restricciones de números visibles y total de minas.
2. Deducciones de conjuntos vacíos/llenos y diferencias entre subconjuntos.
3. Enumeración exhaustiva cuando quedan como máximo 12 celdas de frontera incierta. Pondera cada asignación por el número combinatorio de maneras de distribuir minas fuera de la frontera; las probabilidades son exactas bajo el generador uniforme de minas condicionado por las pistas observadas.
4. Si la frontera supera el límite, conserva deducciones seguras y usa **densidad global como heurística**, eligiendo al azar entre candidatos con el mismo valor. Esa estimación no es una probabilidad posterior exacta y puede omitir deducciones.

La clausura de restricciones está acotada por rondas y tamaño; se limita la generación de diferencias grandes salvo cuando prueban celdas seguras o minas. Esto puede perder deducciones, pero no vuelve inseguras las deducciones obtenidas. El solver nunca abre minas que haya probado como tales. Puede perder por adivinación.

## Métricas

Se separan victorias automáticas (cero decisiones) y victorias después de acciones. `decision_win_rate` excluye las aperturas que ganan automáticamente; incluye intervalo de Wilson del 95%. `proven_safe_clicks` significa deducción demostrada, no simplemente clic que sobrevivió. `guesses` y `successful_guesses` distinguen decisiones inciertas y su supervivencia. Para el jugador aleatorio los clics se marcan `random`, sin gastar cómputo en clasificarlos como deducciones. `mean_safe_fraction` es la fracción de celdas seguras reveladas al terminar, incluida la apertura. Se guardan resultados pareados y detalles de cada partida.

## Conectar la escena sin confundir agentes

`reference/results/solver-replay.json` es una partida con `actor: "solver"`, tamaño, minas públicas y una secuencia de eventos: tablero visible anterior/posterior, acción, clasificación lógica, estimación de riesgo y resultado. `neural_activity` es `null`: **no generar actividad cerebral ficticia para una partida del solver**. La escena puede usar el mismo formato con `actor: "fly"` cuando reciba decisiones y actividad reales del modelo. Los índices de acciones son fila×tamaño+columna, no el canvas de 16×16 del modelo.

Para inspeccionar una decisión neuronal futura: convertir su índice de canvas con `curriculum.decode`, luego llamar `analyze` sobre `env.observation()` y `env.legal_mask()`, y `analysis.classify(action)`. Si devuelve `guess`, significa que este solver acotado no ha probado seguridad, no que sea imposible deducirla.
