# Comparación de entrada actual y local3x3

Experimento: runs/spatial-diagnostic-001; ejecutable experiments/spatial_probe.py. El híbrido permanece pausado, sin cambios de pesos.

Comparación acotada en dos semillas emparejadas,200actualizaciones por variante, batch16 y mismo orden de minibatches por semilla. Ganancias cerebrales iniciales cero y cabezas nuevas idénticas dentro de cada par; se reutiliza únicamente el mapa fijo de entrada/salida. Cerebro completo y cabeza entrenados con la misma pérdida supervisada, Adam y tasas. No se congelan ganancias. Orden de ejecución invertido en la segunda semilla.

Actual: one-hot por celda. Local: misma dimensionalidad, suma del one-hot central y0.5veces promedio de categorías en los8vecinos, máscara de padding y normalización de energía por tablero. Codificador FIJO, sin parámetros adicionales ni información oculta. Se trata de una representación local específica, no de una CNN aprendida ni una prueba de todas las entradas espaciales.

Dataset:256posiciones seguras (128por dificultad5x5/3,7x7/7),128posiciones reservadas (64por dificultad),128tableros completos nuevos (64por dificultad). Una posición por tablero, seleccionada entre estados con oportunidad segura de una trayectoria del solver; incluye diferentes momentos del juego. Los tres conjuntos son disjuntos por hash de distribución real de minas. La verdad oculta se usa solo para evitar repetir tableros entre conjuntos, nunca en las entradas, etiquetas o selección de acciones de los modelos. Etiquetas con todas las casillas demostrablemente seguras según pistas públicas. Archivos de validación registran separación y legalidad.

Medidas: aciertos en posiciones conocidas y reservadas, pérdida, ganancias cambiadas, victorias sin aperturas automáticamente ganadas en partidas autónomas. Conservar respuestas por tablero para comparar de forma emparejada; no sumar dos semillas sobre los mismos tableros como muestras independientes. Las posiciones de prueba visitadas por el solver y las trayectorias autónomas son distribuciones distintas. Presupuesto exploratorio y solo dos semillas: incluso una ventaja observada no autoriza declarar superioridad general.
