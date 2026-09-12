# Estado del proyecto — 12 de septiembre de 2026

## Implementado y verificado

- Entorno Buscaminas 5×5 con 3 minas y apertura central segura automática.
- MaleCNS v1.0 descargado desde Janelia/Google Cloud: 1,109,008,094 bytes.
- Integridad contrastada con MD5 del servidor; SHA-256 en `data/processed/provenance.json`.
- 166,700 neuronas; 25,582,938 conexiones entre neuronas retenidas; 124,177,617 contactos.
- Modelo simplificado de tasas con ganancias internas entrenables y salida entrenable.
- Seis tests pasan: reglas, separación de estado oculto, gradientes numéricos y checkpoint.

## Piloto completado

Corrida `runs/pilot-001`: 128 episodios, 460 decisiones del agente, 125 actualizaciones.
Tres episodios terminaron ya con la apertura automática y no actualizaron el modelo.
Duración total incluyendo evaluaciones: 65.86 segundos. Pico de RAM del proceso
reportado por macOS: 543,162,368 bytes (~0.54 GB). CPU local, sin GPU.
La memoria de importación no está incluida en ese pico.

Evaluación estocástica en 32 tableros separados del entrenamiento:

| Política | Victorias | Casillas seguras reveladas, promedio |
|---|---:|---:|
| Aleatoria | 1/32 | 15.31 |
| Red inicial | 2/32 | 15.38 |
| Tras 128 episodios | 0/32 | 16.69 |

No hay mejora demostrada en victorias. Una muestra tan pequeña no establece tampoco
una degradación estadísticamente concluyente. Cambiaron 165,163 ganancias internas;
parámetros y gradientes finitos. Eso verifica entrenamiento, no aprendizaje del juego.

## Continuación del piloto completada

La continuación de 2,048 episodios terminó: 2,176 en total, checkpoint final
`checkpoint-002176.npz`. Duró 841.29 segundos y ganó 9/32 partidas de evaluación
(28.125%), frente a 2/32 de la red inicial. Son resultados preliminares con una
semilla y muestra pequeña; todavía no establecen una ventaja del conectoma.

- PID y comando: `runs/pilot-001/process.json`.
- Salida: `runs/pilot-001-continuation.log`.
- Métricas: `runs/pilot-001/metrics.jsonl`.
- Último checkpoint completo: el indicado por `runs/pilot-001/state.json`.
- Configuración y hashes de código: `runs/pilot-001/config.json`.
- Copia del código usado: `runs/pilot-001/source/`.

Para detenerlo, verificar primero que el PID registrado corresponda a este entrenador;
enviar SIGTERM a ese proceso. Se conserva el último checkpoint de cada 32 episodios.
Para reanudar, usar el comando de README. No iniciar otro entrenamiento simultáneo
sin comprobar antes si terminó la continuación.

## Nuevo currículo autorizado

El usuario pidió tableros variados y grandes, mezcla de fáciles/difíciles con decay
hasta solo grandes, y más tiempo y partidas. Se lanzó `runs/curriculum-001` con
50,000 partidas nuevas, partiendo del checkpoint final anterior, que no se sobrescribe.

- Inicio: 60% pequeños (5×5/7×7), 30% medianos (9×9/12×12), 10% grandes (16×16).
- Decay lineal de pequeños/medianos a cero durante las primeras 35,000 partidas.
- Últimas 15,000: solamente 16×16, variando entre 31 y 56 minas.
- Lienzo máximo 16×16 con máscara de acciones y padding distinto de casillas tapadas.
- Migración conserva ganancias internas y comportamiento 5×5 inicial dentro de
  tolerancia numérica; entradas adicionales solo fuera del 5×5 y salidas nuevas.
- Se reinicia Adam; recompensa normalizada por cantidad de casillas seguras y gamma=1.
- Se recomputan activaciones para limitar RAM. Checkpoints atómicos cada 256 episodios.
- Evaluación final: 192 tableros nuevos distribuidos en seis estratos; políticas final,
  inicial migrada y aleatoria, con tasas reportadas separadamente por dificultad.
- Fuente: `train_curriculum.py`, `curriculum.py`, extensión compatible de `brain.py`.
- Once tests pasan. Prueba completa del conectoma en `runs/curriculum-smoke-001`;
  completó 32 partidas en dos invocaciones, restauró el currículo y verificó fase final
  solo grande: 28.95 s de entrenamiento, 150 clics, ~0.56 GB RAM. Su evaluación de dos
  tableros por estrato solo verifica ejecución, no rendimiento.

Corrida larga lanzada con PID inicial 89204 (verificar que siga correspondiendo al
comando), registro `runs/curriculum-001-process.json`, log `runs/curriculum-001.log`.
Estimación extrapolada de la prueba: 12–15 horas; depende de la duración futura de
partidas. Se creó el seguimiento `avisar-al-terminar-buscaminas`, cada 15 minutos,
silencioso mientras avance y con aviso al finalizar o fallar. No inicia corridas nuevas.

Consultar `runs/curriculum-001/progress.json`, `state.json` y `live-process.json` para
avance real. `completed.json` certifica fin de corrida y evaluación; no inferirlo
solo de un PID ausente. Los comandos para iniciar/reanudar están en README.

## Pendiente

Evaluar el currículo, ampliar evaluación con nuevas semillas, probar cerebro
congelado y modelos de control antes de afirmar ventaja del conectoma. El diseño visual
está en `DISENO.md`; la visualización 3D local ya está disponible en `scene/`. La actividad del modelo
actual son tasas artificiales, no spikes ni una reproducción fisiológica completa.

## Corrección del visor: activaciones y partidas

El visor local `http://127.0.0.1:8765/` muestra ahora 18 partidas de semillas
prefijadas del checkpoint 22,528. Conserva todos los resultados: 1/6 victorias
5×5, 0/6 9×9 y 0/6 16×16; muestra ilustrativa pequeña, no evaluación final.
Se corrigió el bucle que repetía una misma derrota: avanza entre tableros y
se detiene al terminar la lista. Cada clic incluye los tres estados internos
reales del cálculo de tasas, con magnitud representada por tamaño y brillo
y escala fija (saturación visual en |h|=0.1). No son spikes biológicos.
Verificados continuidad de tableros, legalidad de acciones, estados de actividad,
transiciones de reproducción y carga del control WebMCP; sin errores en el
registro del navegador. Entrenamiento no modificado por esta corrección.

## Escena realista e investigación de entrenamiento

Se reemplazaron muebles Kenney por Metal Office Desk y Modern Arm Chair 01 de
Poly Haven, con texturas 2K, HDRI de estudio y sombras. Monitor, teclado, mouse
y torre son Omie's Office Set CC0; archivos compartidos evitan duplicar texturas.
Fuentes y hashes: `scene/dist/assets/realistic/`. La mosca Flybody se conserva.
Buscaminas clásico tanto en el monitor como en el panel: biseles grises, colores
por número, carita y contadores. El reloj es tiempo de reproducción; no rendimiento
del entrenamiento. Apertura central segura explicada en la página.

`scene/dist/leg-ik.js` articula la pata delantera derecha original con CCD,
sin estirar segmentos; cursor, mouse y pata comparten recorrido interpolado.
`node scene/test-leg-ik.mjs` verifica 101 posiciones, error máximo <0.001 unidades
de escena. Sintaxis y referencias de texturas/buffers verificadas; visor HTTP 200,
control WebMCP registrado y sin errores de navegador en la comprobación de carga.
No se afirma validación biomecánica ni aprendizaje motor.

`research/METODOS_ENTRENAMIENTO.md` compara FlyGM (imitación+PPO), FLYNN (DAgger),
Flybody, DOOMFLY y Minecraft con fuentes primarias. Propone imitación del solver
visible, DAgger y después RL, como nueva corrida futura; no altera el currículo activo.

## Implementación de DAgger, QR-DQN y combinado

Entrenadores independientes en `experiments/train.py`; no se modificaron `brain.py`
ni `train_curriculum.py`. Los tres actualizan ganancias del conectoma completo y
una cabeza dueling de cuantiles. DAgger usa etiquetas de jugadas equivalentes;
QR-DQN utiliza Double, PER con corrección de importancia, n-step y red objetivo;
el híbrido suma imitación y TD con señales separadas. No es Rainbow exacto: no NoisyNet.

Entradas públicas adicionales en la cabeza: tamaño y densidad total de minas,
sin mapa oculto. Estado/replay compacto, grafo compartido, batch16, sin render.
Checkpoints atómicos con pesos/Adam/target/replay/RNG; límite por invocación
`--stop-after` y `--resume` mantienen calendarios. Siete pruebas de gradientes,
máscaras y replay pasan. Smoke real de hybrid hizo cuatro updates; smoke DAgger
reanudado 2+2 coincide exactamente con cuatro episodios continuos (versión inicial,
antecedente al contexto público añadido; registros preservados).

Pilotos `runs/method-pilot-001/{dagger,qrdqn,hybrid}`: 64 episodios por método,
ocho evaluaciones por estrato, sin maestro ni exploración, semillas 2,000,000,000.
Supervisor secuencial: `experiments/run_suite.py`; no confundir estos pilotos con
entrenamientos largos ni con aprendizaje demostrado. Consultar `suite-state.json`
y `comparison.json` para el resultado real. Configuración y copias de fuentes por corrida.

Investigación y mediciones: `research/OPTIMIZACIONES_ENTRENAMIENTO.md` y `benchmarks/`.
Backbone batch16 forward+backward medido 1.41× frente a serial; B32/B64 fueron peores
por muestra. No sumar este factor al 3.07× previo ni extrapolarlo al algoritmo completo.

Los tres pilotos method-pilot-001 finalizaron con checkpoints y evaluación completa.
Updates DAgger/QR-DQN/híbrido: 69/55/88; clics 307/248/381. Evaluación 48 tableros
por método: 1/48, 1/48 y 0/48 respectivamente, sin victorias automáticas. No permite
ordenar métodos: entrenamiento y muestra muy pequeños. Informe `experiments/PILOTOS.md`.
26 tests pasan. Verificación del híbrido actual: dos episodios con cuatro hilos y
reanudación de otros dos con uno coinciden exactamente con cuatro continuos, incluido
replay, Adam, pérdidas y RNG. El backend nuevo predetermina cuatro hilos con particiones
CSR compartidas: prueba B16 1.593→0.877 s forward/backward, igualdad exacta. Tiempos de
los primeros pilotos son de un hilo. No se lanzaron las tres corridas largas de 50,000.
