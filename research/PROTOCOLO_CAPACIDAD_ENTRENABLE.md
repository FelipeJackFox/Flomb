# Protocolo del piloto de capacidad

Autorización: comparar un control convencional con una variante más expresiva del conectoma antes de otra corrida larga. El híbrido queda pausado y no se migra su checkpoint a una arquitectura incompatible.

## Modelos

- `current`: grafo completo y dinámica original de tres propagaciones tanh; una ganancia por neurona, mapas fijos y cabeza lineal. La cabeza representa directamente logits, con la misma clase funcional de la media de cuantiles original. No es una reproducción del entrenamiento QR-DQN.
- `expressive`: mismo grafo y mapas, codificador espacial convolucional entrenable, dos canales por neurona, ganancias, sesgos internos, mezcla compartida de canales, retención aprendida y reinyección de entrada. No es una réplica de FlyGM ni un modelo biológico de spikes.
- `cnn`: control de tres capas convolucionales con lectura por casilla, pistas visibles y contexto público de tamaño/densidad. No reemplaza el objetivo de entrenar el conectoma.

## Datos y comparación

512 posiciones de entrenamiento obtenidas de recorridos del maestro, una por tablero. Solo se incluyen estados con seguridad certificada; se consideran correctas todas las acciones seguras etiquetadas. La pérdida utiliza distribución uniforme sobre ese conjunto, como el entrenador anterior.

126 posiciones reservadas después de retirar dos estados visibles repetidos respecto al entrenamiento, y 48 tableros independientes de partidas completas. Se usan hashes del mapa oculto exclusivamente para impedir contaminación entre conjuntos. El maestro recibe únicamente estado visible, máscara legal y total público de minas.

Entrenamiento equilibrado entre cuatro grupos: tamaño 5×5/7×7 y deducción elemental/relacional o exacta. El segundo grupo significa que el solver encuentra seguridad que un cierre simple de reglas cero/completo no detecta. No demuestra que el modelo utilice una regla particular.

Dos semillas, 120 actualizaciones, minibatches emparejados de 16 y Adam a 0.001. Sin búsqueda de hiperparámetros. Igualar ejemplos y actualizaciones no iguala parámetros, costo ni dificultad de optimización. La variante expresiva cambia varios componentes; cualquier mejora requerirá ablaciones posteriores.

El piloto registra aciertos y pérdida antes/después, probabilidad asignada a acciones seguras, aciertos por estrato, victorias autónomas excluyendo apertura automática, parámetros modificados y segundos de entrenamiento. La memoria es el máximo acumulado del proceso macOS, no memoria por modelo. No se seleccionan checkpoints mirando el conjunto reservado.

## Verificaciones

Seis pruebas nuevas: equivalencia de features/gradiente con el modelo actual; gradiente numérico del operador disperso; gradientes de encoder, sesgos, ganancias, mezcla y retención; restauración exacta de pesos/Adam; máscara y acciones equivalentes; separación de tableros/estados visibles. Siete pruebas existentes de backbone y QR también pasan.

La multiplicación dispersa conserva el backend SciPy de cuatro particiones balanceadas. PyTorch aporta autodiferenciación de los componentes nuevos. No se poda el grafo ni se reutilizan features obsoletas. Checkpoints, manifiesto, hashes y copias del código se guardan por piloto. Se comprueba SHA256 del híbrido antes y después.

## Reproducción

Desde la raíz del proyecto:

```sh
.venv/bin/python -m pip install -r experiments/requirements-capacity.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest experiments.test_expressive_models experiments.test_backbone experiments.test_qr_core
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.capacity_probe --out runs/capacity-diagnostic-NUEVO
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.capacity_references runs/capacity-diagnostic-NUEVO
.venv/bin/python -m experiments.report_capacity_probe runs/capacity-diagnostic-NUEVO
```

Usar un directorio nuevo. La ejecución rechaza sobrescribir un manifiesto existente. Los archivos `.pt` son de este diagnóstico y no se cargan directamente en `experiments/train.py`.

## Decisión posterior

Una mejora en posiciones aisladas debe acompañarse de progreso en partidas nuevas antes de lanzar entrenamiento largo. Para atribuir valor a la anatomía faltará un control con conectividad reconfigurada. Dos posiciones relacionales 7×7 reservadas no permiten una conclusión por ese estrato; habrá que ampliar esa evaluación. Los datos de este piloto tampoco enseñan apuestas probabilísticas: esa parte requiere otro conjunto y métricas de riesgo explícitas.
