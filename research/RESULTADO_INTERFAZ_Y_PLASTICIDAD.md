# Interfaz por casilla y plasticidad de conexiones

Semilla 20260926; 3000 actualizaciones, batch 16 y minibatches idénticos. 10000 posiciones de entrenamiento; 1000 de validación; 500 partidas finales nuevas.

Se reutilizan entrenamiento/validación de la fase ampliada. Las partidas finales no comparten layouts con esos conjuntos ni con los pilotos espacial y de capacidad anteriores. Varias posiciones de validación pertenecen a un mismo tablero; no son ensayos independientes. La comparación tiene una semilla y no adopta modelos automáticamente.

| Modelo | Aciertos reservados antes → después | 5×5 victorias sin apertura automática | 7×7 victorias | Tiempo de entrenamiento |
|---|---:|---:|---:|---:|
| cnn | 391 → 970/1000 | 159/228 (69.7%; IC95% 63.5%–75.3%) | 109/250 (43.6%; IC95% 37.6%–49.8%) | 0.31 min |

En curso. Estado al generar el informe: {'phase': 'training', 'variant': 'retina_fixed', 'seed': 20260926, 'updates': 200, 'target': 3000, 'loss': 2.327831506729126, 'gradient_norm': 0.4682052433490753, 'training_seconds': 164.7006565000047}.

## Uso de decisiones seguras y plasticidad

| Modelo | Elecciones seguras / oportunidades | Muertes con opción segura | Ganancias de conexiones modificadas | Ganancias neuronales modificadas |
|---|---:|---:|---:|---:|
| cnn | 2719/2870 | 77 | — | — |

## Intervenciones sin reentrenar

| Modelo | Intervención | Aciertos reservados | 5×5 victorias | 7×7 victorias |
|---|---|---:|---:|---:|

## Qué prueba y qué no prueba

- Los 166,700 nodos y 25,582,938 conexiones permanecen en las variantes del conectoma. Todas las conexiones originales tienen una ganancia independiente entrenable en retina_plastic. Que una ganancia exista no implica que reciba gradiente en cada minibatch: el informe cuenta las realmente modificadas.
- Entrada en 4426 neuronas L1/L2/L3 usando columnas ópticas anotadas. La lectura por casilla interpola poblaciones Mi/Tm/T1/C3 y usa el mismo decoder en todas las posiciones. La entrada y lectura usan poblaciones distintas. El decoder recibe estado neuronal y contexto público constante por tablero, no pistas directas.
- El mapeo de hexágonos a una pantalla cuadrada, la imagen idéntica para ambos ojos y los tres pasos de tasas son decisiones de ingeniería, no fisiología validada.
- retina_fixed mantiene los pesos de conexiones fijos, pero aprende ganancias por neurona e interfaces. retina_plastic añade 25.6 millones de multiplicadores independientes, positivos y acotados: conserva signos y topología.
- El grafo de control aceptó 13,310,849 intercambios. Cambió 63.8% de las asignaciones fuente-peso originales. Conserva grados de entrada/salida, multiconjunto de pesos entrantes, signos por conexión y autoconexiones existentes. No conserva grados ponderados de salida ni constituye una aleatorización completa.
- Anular toda la actividad produce scores iguales por casilla por construcción. La caída bajo zero solo verifica dependencia funcional del canal neuronal; no prueba una ventaja del conectoma biológico. Shuffle permuta la actividad entre neuronas sin cambiar su distribución global.
- La comparación entrenada con rewired_plastic, con iguales interfaces y presupuesto, es el control para empezar a evaluar el aporte del cableado. Requiere réplicas antes de atribuir una ventaja a la anatomía.
- No se reanuda el híbrido anterior ni se inicia DAgger automáticamente. Los datos supervisados aún cubren decisiones seguras certificadas, no un objetivo explícito de apuestas.
