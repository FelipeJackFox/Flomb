# Flomb — una mosca que aprende buscaminas

### ▶ Demo en vivo: **https://overfitters.space/proyectos/moscas/buscaminas** · [English](https://overfitters.space/en/proyectos/moscas/buscaminas) · proyecto del [LEIA](https://overfitters.space/proyectos/moscas)

*Proyecto de Felipe ([@FelipeJackFox](https://github.com/FelipeJackFox)) dentro del LEIA, donde cada integrante desarrolla el suyo. Nombre de trabajo anterior: "Mosca Fruta".*

Una mosca simulada, con el cuerpo pedunculado del conectoma MaleCNS (4,064 células de Kenyon, 97 neuronas de salida, dopamina PAM/PPL1), aprende a jugar buscaminas con dos únicas señales: **calor** cuando pisa una mina y **azúcar** cuando pisa una casilla segura. Sin backprop, sin red que piense antes, sin lector entrenado después.

**Estado actual (4 oct 2026).** Una sola mosca gana **78.4%** de las partidas 9×9 con 12 minas (IC95 77.1–79.7; 30 semillas, 2,000 tableros nunca vistos, inicio estándar con apertura en cascada). Una política de "un solo punto" escrita a mano gana 81.8% en esos mismos tableros. En 7×7 gana 89.1% y en 16×16, 59.2%. Corrida: `runs/fly-replica-001`.

- **Sentidos crudos.** Al antenar cada vecina abierta huele tres cantidades: N (la pista), k (cuánto tapado la rodea) y m (cuántas marcas propias la rodean). La arena no divide ni resta nada por ella.
- **Aprende sola cuándo marcar.** El umbral de "esto es una mina" lo descubre con recompensa retrasada; quedó junto al valor que se había puesto a ojo.
- **Qué aporta el cerebro:** aprende con ~4× menos experiencia que una tabla sin cerebro. Con experiencia ilimitada la tabla empata.
- **Qué no aporta:** el cableado concreto de la mosca. Barajarlo da lo mismo; sirve la arquitectura (expansión dispersa + lectura plástica con dopamina).

| Para… | Ver |
|---|---|
| El resultado completo, con lo que no funcionó | [`research/REPORTE_FINAL_MOSCA.md`](research/REPORTE_FINAL_MOSCA.md) |
| Qué es un sentido legítimo y qué es trampa | [`research/ANALISIS_SENTIDOS_MOSCA.md`](research/ANALISIS_SENTIDOS_MOSCA.md) |
| La bitácora, experimento por experimento | [`ESTADO.md`](ESTADO.md) (se lee por el final) |
| El código de la mosca | `experiments/mushroom_body.py`, `experiments/fly_senses.py`, `experiments/fly_agents.py` |
| Cada corrida formal | `runs/fly-*/summary.json` y su protocolo en `research/PROTOCOLO_*.md` |
| El visor local 3D + 2D | `scene/dist/arena.html` (servir con `scene/serve.py`) |

Los números de corridas distintas usan tableros distintos; las comparaciones limpias son las internas de cada corrida. Es una arquitectura inspirada en el cuerpo pedunculado que usa sus conteos reales de sinapsis (modelo de tasas, sin spikes), no una simulación de una mosca.

Datos: conectoma [MaleCNS v1.0](https://male-cns.janelia.org/download/) (CC-BY). Modelo anatómico: Flybody (Google DeepMind + HHMI Janelia, Apache-2.0).

---

## Historia anterior del repositorio (sept 2026, línea con backprop)

Lo que sigue es el README original de la primera etapa: entrenar por refuerzo y por backprop alrededor del grafo completo. Esa línea llegó a ~9.8% en 9×9 y se abandonó; se conserva como registro.

### Mosca Fruta: MaleCNS y Buscaminas

## Currículo de 50,000 partidas

La corrida nueva `runs/curriculum-001` parte de las ganancias y salida del checkpoint
de 2,176 partidas. El original se conserva intacto. La entrada/salida pasa a un
lienzo de 16×16: el antiguo 5×5 ocupa la esquina superior izquierda y la nueva
entrada solo añade señales de las celdas exteriores. Con un tablero 5×5, antes de
seguir entrenando, las probabilidades originales se conservan dentro de tolerancia numérica.
La región fuera del tablero es todo ceros y no admite acciones; una celda tapada
real tiene su propio canal one-hot. Se reinician los momentos de Adam al migrar.

| Grupo | Tamaños | Densidad aproximada de minas | Mezcla inicial |
|---|---|---|---:|
| Pequeños | 5×5, 7×7 | 8–16% | 60% |
| Medianos | 9×9, 12×12 | 10–19% | 30% |
| Grandes | 16×16 | 12–22% (31–56 minas) | 10% |

Las probabilidades de pequeños y medianos decaen linealmente a cero durante las
primeras 35,000 partidas. Las últimas 15,000 son exclusivamente 16×16, con densidad
variable. El progreso del currículo se restaura del checkpoint, sin reiniciar el decay.

Se modifica también la recompensa: −1 por mina, +1 por victoria y un bono total
máximo de 0.25 por revelar casillas seguras; descuento gamma=1. Así los tableros
grandes no reciben más recompensa simplemente por abrir muchas casillas. Cada
grupo mantiene su propio baseline. Se recomputan las activaciones para el backward,
evitando guardar todo el cerebro por cada clic de una partida larga.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python train_curriculum.py --run runs/curriculum-001 --total-episodes 50000 --chunk-episodes 50000 --eval-per-stratum 32
```

Reanudar, después de verificar que no haya otro proceso activo:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python train_curriculum.py --run runs/curriculum-001 --resume --total-episodes 50000 --chunk-episodes 50000 --eval-per-stratum 32
```

`--total-episodes` es el presupuesto fijo del currículo; `--chunk-episodes` limita
esa ejecución y se recorta al total. Hay bloqueo de escritura concurrente y
checkpoints atómicos cada 256 partidas. Los logs de invocaciones interrumpidas se
conservan por separado. `progress.json` es el avance más reciente; `state.json`
identifica el último checkpoint completo. `completed.json` solo aparece tras guardar
el checkpoint final **y** terminar la evaluación.

Se evalúan seis estratos: 5×5/3, 7×7/7, 9×9/10, 12×12/24, 16×16/40 y 16×16/56.
La evaluación final usa 32 tableros nuevos por estrato (192 total), separados tanto
de entrenamiento como de la evaluación inicial, y compara la política final,
la inicial migrada y una aleatoria en esos mismos tableros. No seleccionar resultados
solo por el promedio: revisar cada tamaño y dificultad, y las victorias automáticas.

```sh
.venv/bin/python -m unittest -v test_core test_curriculum
```

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
