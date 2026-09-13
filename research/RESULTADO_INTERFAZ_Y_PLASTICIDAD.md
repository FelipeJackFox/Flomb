# Interfaz por casilla y plasticidad de conexiones

Semilla 20260926; 3000 actualizaciones, batch 16 y minibatches idénticos. 10000 posiciones de entrenamiento; 1000 de validación; 500 partidas finales nuevas.

Se reutilizan entrenamiento/validación de la fase ampliada. Las partidas finales no comparten layouts con esos conjuntos ni con los pilotos espacial y de capacidad anteriores. Varias posiciones de validación pertenecen a un mismo tablero; no son ensayos independientes. La comparación tiene una semilla y no adopta modelos automáticamente.

| Modelo | Aciertos reservados antes → después | 5×5 victorias sin apertura automática | 7×7 victorias | Tiempo de entrenamiento |
|---|---:|---:|---:|---:|
| cnn | 391 → 970/1000 | 159/228 (69.7%; IC95% 63.5%–75.3%) | 109/250 (43.6%; IC95% 37.6%–49.8%) | 0.31 min |
| retina_fixed | 324 → 789/1000 | 73/228 (32.0%; IC95% 26.3%–38.3%) | 9/250 (3.6%; IC95% 1.9%–6.7%) | 38.47 min |
| retina_plastic | 324 → 780/1000 | 68/228 (29.8%; IC95% 24.3%–36.1%) | 11/250 (4.4%; IC95% 2.5%–7.7%) | 42.28 min |
| rewired_plastic | 377 → 826/1000 | 90/228 (39.5%; IC95% 33.4%–45.9%) | 17/250 (6.8%; IC95% 4.3%–10.6%) | 46.21 min |

## Uso de decisiones seguras y plasticidad

| Modelo | Elecciones seguras / oportunidades | Muertes con opción segura | Ganancias de conexiones modificadas | Ganancias neuronales modificadas |
|---|---:|---:|---:|---:|
| cnn | 2719/2870 | 77 | — | — |
| retina_fixed | 979/1315 | 221 | 0 | 55464 |
| retina_plastic | 1098/1441 | 223 | 2106242 | 55464 |
| rewired_plastic | 1355/1661 | 216 | 8682974 | 108923 |

## Intervenciones sin reentrenar

| Modelo | Intervención | Aciertos reservados | 5×5 victorias | 7×7 victorias |
|---|---|---:|---:|---:|
| retina_fixed | zero | 388/1000 | 5/228 (2.2%; IC95% 0.9%–5.0%) | 0/250 (0.0%; IC95% 0.0%–1.5%) |
| retina_fixed | shuffle | 396/1000 | 11/228 (4.8%; IC95% 2.7%–8.4%) | 0/250 (0.0%; IC95% 0.0%–1.5%) |
| retina_plastic | zero | 388/1000 | 5/228 (2.2%; IC95% 0.9%–5.0%) | 0/250 (0.0%; IC95% 0.0%–1.5%) |
| retina_plastic | shuffle | 415/1000 | 11/228 (4.8%; IC95% 2.7%–8.4%) | 1/250 (0.4%; IC95% 0.1%–2.2%) |
| rewired_plastic | zero | 388/1000 | 5/228 (2.2%; IC95% 0.9%–5.0%) | 0/250 (0.0%; IC95% 0.0%–1.5%) |
| rewired_plastic | shuffle | 396/1000 | 8/228 (3.5%; IC95% 1.8%–6.8%) | 0/250 (0.0%; IC95% 0.0%–1.5%) |

## Qué prueba y qué no prueba

- Los 166,700 nodos y 25,582,938 conexiones permanecen en las variantes del conectoma. Todas las conexiones originales tienen una ganancia independiente entrenable en retina_plastic. Que una ganancia exista no implica que reciba gradiente en cada minibatch: el informe cuenta las realmente modificadas.
- Entrada en 4426 neuronas L1/L2/L3 usando columnas ópticas anotadas. La lectura por casilla interpola poblaciones Mi/Tm/T1/C3 y usa el mismo decoder en todas las posiciones. La entrada y lectura usan poblaciones distintas. El decoder recibe estado neuronal y contexto público constante por tablero, no pistas directas.
- El mapeo de hexágonos a una pantalla cuadrada, la imagen idéntica para ambos ojos y los tres pasos de tasas son decisiones de ingeniería, no fisiología validada.
- retina_fixed mantiene los pesos de conexiones fijos, pero aprende ganancias por neurona e interfaces. retina_plastic añade 25.6 millones de multiplicadores independientes, positivos y acotados: conserva signos y topología.
- El grafo de control aceptó 13,310,849 intercambios. Cambió 63.8% de las asignaciones fuente-peso originales. Conserva grados de entrada/salida, multiconjunto de pesos entrantes, signos por conexión y autoconexiones existentes. No conserva grados ponderados de salida ni constituye una aleatorización completa.
- Anular toda la actividad produce scores iguales por casilla por construcción. La caída bajo zero solo verifica dependencia funcional del canal neuronal; no prueba una ventaja del conectoma biológico. Shuffle permuta la actividad entre neuronas sin cambiar su distribución global.
- La comparación entrenada con rewired_plastic, con iguales interfaces y presupuesto, es el control para empezar a evaluar el aporte del cableado. Requiere réplicas antes de atribuir una ventaja a la anatomía.
- No se reanuda el híbrido anterior ni se inicia DAgger automáticamente. Los datos supervisados aún cubren decisiones seguras certificadas, no un objetivo explícito de apuestas.
