# Mosca Fruta: MaleCNS y Buscaminas

Primer experimento local de aprendizaje por refuerzo sobre el grafo neuronal
MaleCNS v1.0. El diseño visual acordado —mosca sentada frente a una computadora,
tablero y actividad cerebral 3D— queda para una etapa posterior. Aún no está implementado.

## Experimento inicial

- Buscaminas 5×5, tres minas, apertura central segura automática y expansión de ceros.
- Observación: solamente las 25 casillas visibles, codificadas como tapada o número 0–8.
- Acciones: descubrir una casilla tapada; sin banderas en esta primera versión.
- Recompensa: −1 por mina, +1 por victoria y +0.05 por casilla segura descubierta.
- Grafo: todas las conexiones publicadas entre neuronas con superclass asignada,
  excluyendo status Glia, sin umbral adicional de fuerza. Se conservan auto-conexiones.
- Dinámica artificial de tasas: tres propagaciones tanh por decisión; estado reiniciado
  en cada decisión. No se simulan voltajes, spikes ni milisegundos biológicos.
- Entrada: proyección aleatoria fija del tablero sobre las neuronas; salida desde
  1024 neuronas receptoras seleccionadas al azar. Son interfaces de ingeniería,
  no retina ni circuitos motores biológicamente calibrados.
- Se entrena una ganancia positiva por neurona presináptica y una salida a 25 acciones.
  Cada ganancia modifica conjuntamente las conexiones salientes de esa neurona;
  no son 25 millones de pesos entrenables independientes. La topología permanece fija.
- Pesos iniciales: cantidad de contactos, signo aproximado por neurotransmisor y
  normalización por suma absoluta de entradas. GABA, glutamato e histamina negativos;
  otros o desconocidos positivos. Esta convención no representa receptores individuales.
- Algoritmo: REINFORCE episódico con retorno descontado, baseline móvil, entropía y Adam.
  Las derivadas de la red dispersa están implementadas en NumPy/SciPy y verificadas
  por diferencias finitas. No se necesita GPU ni autograd para este piloto.

## Ejecutar

Probado con Python 3.14 en macOS ARM. Desde esta carpeta:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_data.py
.venv/bin/python -m unittest -v test_core
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python train.py --run runs/pilot-001 --episodes 128 --eval-episodes 32
```

El preparador descarga aproximadamente 1.11 GB cuando faltan los archivos, verifica
el MD5 contra Google Cloud y registra SHA-256 local. Un archivo `.part` incompleto
debe terminar de descargarse o eliminarse antes de volver a preparar. La importación
tiene mayor consumo temporal de RAM que el entrenamiento. Reservar al menos 5 GB
de disco para entorno, datos y checkpoints iniciales; los checkpoints se acumulan.

Para continuar una corrida existente desde el último checkpoint completo:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python train.py --run runs/pilot-001 --resume --episodes 128 --eval-episodes 32
```

`--episodes` agrega esa cantidad de episodios. Se guarda cada 32 episodios y al terminar,
incluyendo parámetros, estados de Adam, baseline y RNG. Conservar el mismo `--seed`
y `--freeze-internal` al reanudar. Si se interrumpe entre checkpoints, las últimas
filas de métricas posteriores al checkpoint son intentos no consolidados.

Para el control de cerebro congelado usar una carpeta nueva y `--freeze-internal`.
No ejecutar varios procesos con el conectoma completo simultáneamente en esta Mac.

## Evidencia y límites

Cada corrida conserva configuración, checkpoint inicial, métricas por episodio,
evaluación anterior/posterior y checkpoints. Entrenamiento y evaluación usan rangos
de semillas distintos. Las evaluaciones usan política estocástica, los mismos tableros
y la misma semilla de acciones. Se reportan victorias automáticas del primer clic
por separado, además de la tasa de victoria total.

32 partidas de evaluación y una semilla de entrenamiento sirven para verificar
ejecución, no para demostrar aprendizaje ni ventaja del conectoma. Antes de afirmarlo
faltan más entrenamiento, varias semillas, evaluación final nueva y controles de
cerebro congelado, red convencional y conexiones reorganizadas.

Fuentes y licencia de datos:

- [MaleCNS, datos oficiales y licencia CC-BY](https://male-cns.janelia.org/download/)
- [Anuncio Google Research / HHMI Janelia y colaboradores](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)

La procedencia exacta y las transformaciones están en `data/processed/provenance.json`.
Los datos originales, el entorno y las corridas se excluyen de Git; se conservan localmente.
