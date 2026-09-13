# Prueba estructural autorizada

Objetivo: comprobar si una interfaz espacial y plasticidad por conexión permiten aprender mejor, y si la actividad del conectoma y su cableado original aportan al resultado. Híbrido original preservado y pausado.

## Diseño

Cuatro modelos: CNN de control; interfaz óptica con pesos fijos y ganancias neuronales aprendidas (`retina_fixed`); misma interfaz con plasticidad por conexión (`retina_plastic`); plasticidad sobre un grafo reconfigurado (`rewired_plastic`). Una semilla 20260926, 3,000 actualizaciones de batch 16, mismos minibatches balanceados por tamaño y categoría, Adam a 0.001, clip global 5. No búsqueda ni selección de checkpoint usando partidas finales.

Se reutilizan 10,000 posiciones de entrenamiento y 1,000 reservadas de la fase ampliada. Las 500 partidas finales son nuevas, sin compartir layouts con entrenamiento/validación ni con los datasets de los pilotos espacial y de capacidad. Se reservaron 250 por tamaño; no se filtran aperturas difíciles. Las conclusiones comparan modelos en este conjunto; no se atribuyen diferencias con porcentajes de partidas anteriores únicamente al cambio de arquitectura.

## Interfaz

4,426 neuronas L1/L2/L3 reciben tres canales de un encoder convolucional a partir de pistas visibles y contexto público. Sus columnas ópticas anotadas determinan dónde muestrean el tablero. Se proyectan afínmente coordenadas hexagonales a la pantalla cuadrada, mostrando el mismo tablero a ambos ojos: es una interfaz de ingeniería, no visión biológicamente validada.

La salida interpola cuatro vecinos por tipo neuronal y por casilla en diez tipos Mi/Tm/T1/C3. Un decoder compartido asigna una puntuación a cada casilla a partir de esas actividades y del contexto constante tamaño/densidad. No recibe una conexión directa con las pistas del tablero. Entradas y lecturas son conjuntos disjuntos de neuronas.

Se conservan 166,700 nodos y 25,582,938 conexiones en los modelos basados en el grafo original. Cada conexión en la variante plástica recibe un multiplicador independiente exp(log_gain), limitado a exp(-2)..exp(2); conserva el signo inicial. También aprende una ganancia por neurona. No hay garantía de gradiente en cada conexión: las conexiones modificadas se cuentan al finalizar. Tres pasos de propagación, tasas tanh, sin añadir spikes.

## Controles

El grafo reconfigurado conserva los grados de entrada y salida, pesos entrantes por fila, signos y autoconexiones existentes. Se realizaron 25,582,938 intentos y 13,310,849 intercambios aceptados; cambiaron 63.75% de las asignaciones fuente-peso. No conserva fuerza ponderada de salida ni implica aleatorización completa.

Al finalizar cada modelo neuronal se evalúa además con actividad anulada y con actividad permutada entre neuronas. El primer control obliga a scores constantes por casilla por diseño; muestra dependencia funcional, no valor de la anatomía. El grafo reconfigurado entrenado es el control relevante para el cableado, todavía limitado por una semilla.

## Costo, integridad y recuperación

Kernel C++ para derivadas por conexión sin materializar matrices N×N ni E×batch. Omite productos de gradientes exactamente cero, sin aproximar gradientes pequeños ni podar conexiones. Medición real: batch 8 ≈0.111 s/ejemplo; batch 16 ≈0.056 s/ejemplo. Se elige 16; actualizaciones de la variante plástica ≈0.9 s en tres pasos de prueba. Pico acumulado del proceso de preparación ≈2.69 GB; no es memoria aislada de entrenamiento prolongado.

21 pruebas pasan, incluyendo comparación de derivadas con diferencias finitas y cálculo denso, plasticidad independiente entre salidas de una neurona, conservación de grados/signos al reconfigurar, anulación de actividad y recuperación exacta de parámetros y Adam mediante el CLI. Fuentes e inputs quedan identificados por SHA256. Checkpoints atómicos cada 250 actualizaciones, progreso ligero cada 100; reanudar exige el mismo código/configuración/input. `caffeinate` acompaña al proceso para evitar suspensión por inactividad.

Scripts: `prepare_structural.py`, `structural_train.py`, `report_structural.py`, `plastic_sparse.py`, `plastic_kernel.cpp`, `retina_policy.py`. Artefactos: `runs/structural-learning-001`. Informe regenerable: `research/RESULTADO_INTERFAZ_Y_PLASTICIDAD.md`. Al terminar se verifican checkpoints y hash del híbrido; no se adopta una arquitectura ni se lanza otro entrenamiento automáticamente.
