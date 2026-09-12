# Optimización de DAgger, QR-DQN y su combinación

Revisión: 12 de septiembre de 2026. Fuentes primarias consultadas y microbenchmark local. No se inició ni interrumpió ningún entrenamiento durante esta investigación.

## Lo medido aquí

El entrenamiento anterior ya obtuvo 3.07× en una comparación acotada de 16 episodios: partición de filas CSR entre cuatro hilos más reutilización de activaciones **dentro de la misma actualización**. Conservaron parámetros, momentos y RNG idénticos. Ver `benchmarks/ACCELERATION.md`; no extrapolar ese factor a un algoritmo distinto.

Nueva prueba de una multiplicación del grafo real: 166,700 nodos, 25,582,938 conexiones, float32. Mediana de tres repeticiones, mientras otro entrenamiento seguía activo:

| Observaciones por lote | Bucle de vectores | CSR × matriz | Factor bucle / matriz |
|---:|---:|---:|---:|
| 1 | 22.36 ms | 21.76 ms | 1.03× |
| 4 | 86.43 ms | 128.68 ms | 0.67× |
| 8 | 178.36 ms | 241.38 ms | 0.74× |
| 16 | 381.98 ms | 242.62 ms | 1.57× |

Todos los resultados coincidieron exactamente en esta muestra; se comprobó `allclose(rtol=1e-5, atol=1e-6)`, error máximo 0. Pico del proceso: 409.6 MB, ejecución completa unos 6 segundos. Fuente reproducible: `benchmarks/batch_probe.py`, resultados crudos `benchmarks/batch_probe_result.json`.

**Interpretación:** SpMM con lote 16 merece prueba completa; lotes 4 y 8 fueron más lentos. Esto compara un kernel de SciPy sin partición de filas; no compara contra el entrenador actual de cuatro hilos, ni incluye codificación, backpropagation, cabeza QR, replay o guardado. No hay todavía evidencia de otra aceleración global de 1.57×.

## Qué hacen otros proyectos

| Proyecto y evidencia primaria | Optimización documentada | Qué trasladar y qué no |
|---|---|---|
| [FlyGM, método y apéndice A](https://arxiv.org/html/2602.17997v1) | Operador sináptico disperso, codificador de entrada pequeño, actualizaciones neuronales compartidas, imitación antes de PPO, rollouts paralelos y DDP. Reportan servidores A100 de 80 GB; 32 canales neuronales y cuatro capas. | Reutilizar estructura dispersa y buenos ejemplos. DDP es una ruta para varias GPU; no demuestra velocidad equivalente en Mac. Su FlyWire y dinámica multicanal difieren de nuestro MaleCNS tanh de tres pasos. |
| [FLYNN, II-D](https://arxiv.org/html/2607.00025v1) | DAgger con conjunto acumulado; 500 episodios nuevos por iteración durante cuatro iteraciones; muestreo balanceado por cinco conductas y menor control del maestro. | Balancear ejemplos reduce desperdicio de datos repetitivos. El artículo consultado no establece un benchmark de kernels/segundo ni una aceleración CPU trasladable. Su maestro usa información privilegiada; el nuestro debe seguir limitado al tablero visible. |
| [Flybody, script oficial Ray](https://github.com/TuragaLab/flybody/blob/main/flybody/train_dmpo_ray.py) | Configura 32 actores CPU, un learner GPU, servidor replay, batch 256, prefetch 4, retorno de cinco pasos y límite de muestras por inserción. | Separar generación, aprendizaje y evaluación; limitar colas y edad de políticas. No copiar 32 procesos de conectoma en una Mac: consumirían memoria y ancho de banda. Flybody es el cuerpo/entorno y aquí usa DMPO, no prueba de aprendizaje con conectoma. |
| [Ape-X](https://arxiv.org/abs/1803.00933) | Separa actores de aprendizaje, reúne experiencias de actores con exploración diversa en replay priorizado y actualiza con un learner central. | Una inferencia central batched puede evitar réplicas grandes. Solo escalar actores cuando el entorno sea el cuello de botella; Buscaminas es barato y el cerebro domina el costo. |
| [Rainbow](https://arxiv.org/abs/1710.02298) y [QR-DQN](https://arxiv.org/abs/1710.10044) | Rainbow combina varios cambios algorítmicos y muestra su contribución por ablaciones. QR-DQN representa retornos con cuantiles. | Replay y retornos multistep mejoran uso de datos; no equivalen a menor costo por actualización. QR-DQN con PER/dueling/n-step es una variante de la familia, no Rainbow exacto: Rainbow original usa C51 y también NoisyNet. |
| [DOOMFLY, Fast dev / exact clocks](https://fly-brain-doom.awormuth.chatgpt.site/learning) | Omite actualizaciones de navegador, compresión de vídeo y espera de tiempo real; paraleliza réplicas y comprueba equivalencia. Aun así reporta integración neural como 92% del tiempo, 0.16× tiempo real en un registro y 0.21× agregado en otro. | Entrenar sin render y exportar reproducciones aparte ya coincide con nuestra separación. Sus mediciones son del autor en host compartido; no demuestran supervivencia aprendida ni ofrecen un atajo para nuestro kernel. |

## Implementación recomendada y prioridades

1. **Replay compacto del tablero visible.** Guardar estados como 256 enteros pequeños, tamaño y metadatos, y reconstruir one-hot float32 al formar el minibatch. Solo dos tableros int8 por transición ocupan 512 bytes frente a 20,480 bytes de dos observaciones one-hot de 2,560 float32: 40× menos para ese componente. Acciones, prioridades, etiquetas, máscaras y estructura tienen costos adicionales; no afirmar 40× para todo el proceso. Reservar arreglos contiguos y un límite explícito de capacidad.
2. **No guardar activaciones obsoletas.** Si cambian `log_gain` u otros parámetros internos, un tablero antiguo debe atravesar el cerebro actual al volver a muestrearse. Guardar activaciones únicamente hasta aplicar la actualización correspondiente. La red objetivo DQN necesita sus propios parámetros y sus propias activaciones. Cachear features persistentemente solo es correcto con encoder congelado y versionado; congelarlo cambia el experimento, no es una optimización transparente.
3. **Probar lote 16 frente a cuatro hilos y microbatches.** Compartir CSR y su transpuesta; evitar copiar 25.6 millones de conexiones por muestra. Los buffers neuronales escalan como N × B: un estado float32 B16 cuesta 10.67 MB, cuatro estados 42.68 MB, sin contar temporales. Elegir por transiciones/segundo y memoria total, incluyendo ambas redes DQN. No activar todos los hilos en tres procesos simultáneos sin medir contención.
4. **Aprender una vez sobre el conjunto, no repetir forwards innecesarios.** En DAgger puro, calcular etiquetas del solver sin inferencia neural cuando el maestro controla la acción; calcularla cuando la política vaya a decidir o cuando se entrene ese ejemplo. Cachear etiquetas visibles deterministas por tablero y configuración del solver puede ser válido; el teacher no depende de pesos aprendidos. Cualquier dato que dependa de banderas o minas restantes debe formar parte de la clave.
5. **Coste de la cabeza QR.** Calcular cuantiles de todas las acciones para seleccionar, pero loss/gradiente solo de la acción observada; no construir pares de cuantiles para 256 acciones. Con Q cuantiles, la comparación cuantílica seleccionada escala B × Q². Elegir Q y frecuencia de actualización con evaluación, no reducirlos silenciosamente respecto a la configuración registrada.
6. **Control de replay/compute.** Registrar muestras por inserción, número de actualizaciones, consultas al maestro y tiempo de cada fase. Más replay sin límite puede gastar todo el tiempo repitiendo ejemplos. Usar sum-tree u otra estructura incremental para PER evita recalcular una distribución completa en cada muestra; aplicar pesos de importance sampling y prioridades finitas.
7. **Medir pilotos antes de largas corridas.** Mismo conjunto reservado por tamaño/densidad, métrica sin maestro, presupuesto de tiempo y pasos informado; solver-only, modelo inicial y entrenamiento anterior como referencias. En el combinado, hacer explícito el peso de imitación y su decay. No confundir menor loss del maestro con más victorias.

El antecedente directo de combinar demostraciones con Q-learning es [DQfD](https://arxiv.org/abs/1704.03732), que combina TD, retorno multistep y pérdida supervisada sobre demostraciones con replay priorizado. DAgger añade la recolección de etiquetas en estados visitados por el alumno; no transforma etiquetas en transiciones contrafactuales con recompensas inventadas. [DAgger original](https://arxiv.org/abs/1011.0686).

## Optimizaciones que cambian la pregunta

Podar conexiones, reducir neuronas/ciclos, sustituir el cerebro por MLP, congelar ganancias, cuantizar agresivamente o convertir a una dinámica distinta pueden acelerar, pero requieren una variante identificada y evaluación aparte. No presentarlas como el mismo cerebro entrenando más rápido.

Una GPU es candidata si dispone de kernels CSR forward y backward efectivos para este tamaño. Ni los resultados A100 de FlyGM ni tener memoria unificada garantizan aceleración de sparse en Apple Metal. Medir el pipeline completo en el hardware destino antes de mover una corrida; no densificar: la matriz 166,700² float32 sola ocuparía aproximadamente 111 GB decimales.

## Integración medida después de la investigación

Se implementaron los tres entrenadores en `experiments/` y se lanzaron pilotos
secuenciales separados de la corrida histórica. Se midió también forward + backward
completos del backbone: B16 tardó 1.607 s frente a 2.265 s al procesar los 16 estados
individualmente (1.41×). Features idénticas y diferencia máxima de gradiente
5.96e-8. No incluye cabeza, optimizador, solver, replay ni checkpoint; host compartido.

Barrido adicional de batches en la misma máquina: B16 0.0884 s/muestra,
B32 0.1278 y B64 0.1209. No se eligió un lote mayor por intuición: en esta prueba
consumió más memoria y rindió peor por muestra. Se mantiene B16 en los pilotos.
Resultados reproducibles: `benchmarks/backbone_batch_probe.py`, `benchmarks/batch_sizes.py`
y sus JSON. Los números son muestras locales, no un benchmark universal.

Una tercera medición particionó filas CSR del mismo batch16: un hilo 1.593 s,
dos hilos 1.107 s y cuatro hilos 0.877 s. Features y gradientes idénticos. Se integró
`--workers 4` como opción predeterminada para próximas invocaciones; el primer
piloto ya estaba en curso con un hilo. En un híbrido completo se validó reanudación
4→1 hilos frente a ejecución continua, preservando exactamente todo el estado de
aprendizaje y replay. Ver `threaded_batch_probe_result.json` y `resume_equivalence.json`.
