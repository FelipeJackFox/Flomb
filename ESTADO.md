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

## Corrida híbrida principal iniciada · 12 septiembre 2026

- `runs/hybrid-001`: híbrido DAgger + QR-DQN,50,000 episodios, horizonte50,000, batch16,4workers, semilla20260916. PID y comando en launch.json.
- Inicialización común initial-expanded.npz; no es cerebro virgen ni continuación del piloto híbrido. Maestro decae hasta0 en30,000episodios; pérdida de imitación conserva peso mínimo0.1. Evaluación autónoma192tableros cada1,000partidas; métricas en todos los episodios.
- Proceso confirmado produciendo métricas y actualizaciones reales. Currículo manual habilitado entre episodios, sin solicitud manual inicial.
- Selector del dashboard: Corridas principales / Pilotos cortos / Pruebas técnicas. Corrida antigua sigue activa; no se modificó. Seguimiento horario cubre ambas y avisa de finalización o fallo.

## Aceleración híbrido por reparto de conexiones

Reanudado hybrid-001 desde episodio576, proceso actual en live-process.json. Particiones CSR equilibradas por cantidad de conexiones: kernel completo forward/backward1.499× medido, features y gradientes idénticos. No es speedup end-to-end confirmado. Snapshot anterior en runs/hybrid-001/pre-balanced-576. Preservados optimizer, replay, target y RNG; corrida antigua sigue intacta. Detalles benchmarks/BALANCED_PARTITIONS.md.

## Intervención de autonomía aprobada · desde partida817

hybrid-001 reanudado del checkpoint816 con optimizador, replay y redes preservados. Snapshot completo en runs/hybrid-001/pre-behavior-816. behavior-schedule.json y behavior-applied.json registran el calendario: maestro70% en reanudación →30% en3,000 →0% en8,000; exploración condicionada a no usar maestro20% →5% en8,000. Etiquetas del maestro y pérdida de imitación continúan. Actualizaciones posteriores confirmadas; PID vigente en live-process.json y launch.json.

Dashboard: media móvil predeterminada de hasta200episodios por combinación exacta tamaño/minas, agregando contadores antes de dividir. Tooltip n muestra partidas con decisiones; las combinaciones escasas mantienen n pequeño. Vista general suavizada muestra seis dificultades de referencia. Selector permite volver al agregado sin suavizado adicional. Evaluación fija sin suavizar y mezcla observada conservan sus ventanas originales. Seguimiento horario incluye revisión autónoma en3,000 sin cambios automáticos.

## Finalización curriculum-001 notificada · 12 septiembre 2026

Verificados completed.json, checkpoint-050000.npz y evaluation-final.json.50,000episodios completos. Evaluación final:5x5/3=7victorias/31partidas con decisiones (8/32 incluyendo1automática), inicial8/31;7x7,9x9,12x12,16x16/40 y16x16/56:0/32cadauno. No mejora demostrada frente al inicial. Se notifica en la respuesta de aplicación del calendario híbrido; no repetir este aviso en heartbeat. Continúa monitorizando hybrid-001 y revisión3,000.

## Revisión 3,000 hybrid-001 notificada · 2026-09-13 01:26 UTC

Revisión horaria realizada cuando el proceso ya llevaba7,544episodios; PID58381 verificado con comando, checkpoint7,536 y log normal. Evaluaciones autónomas fijas de1,000/2,000/3,000:5x5/3=14/31,17/31,14/31 excluyendo apertura automática; cada dificultad7x7,9x9,12x12,16x16/40,16x16/56=0/32 en las tres. No progreso sostenido entre estos hitos ni generalización demostrada;32tableros por estrato son muestra pequeña. Aviso de revisión emitido una vez; no modificar configuración ni repetir este hito. Continúa seguimiento hasta finalización/fallo.

## Diagnóstico autorizado · híbrido pausado en14,384

Usuario aprobó pausa y pruebas separadas. Checkpoint íntegro respaldado en runs/hybrid-001/pre-diagnostic-14384; paused.json marca pausa intencional y seguimiento horario PAUSED. Prueba de memorización: experiments/diagnose_learning.py → runs/learning-diagnostic-001,100posiciones seguras/100nuevas, pesos del híbrido y Adam reiniciado solo en copia supervisada. DAgger separado: runs/diagnostic-dagger-001,256episodios, mezcla exacta50/50 entre5x5/3 y7x7/7, inicialización común del piloto y cabeza nueva, evaluación autónoma32por estrato. Sin modificaciones adicionales de corrida principal.

## Diagnóstico completado

Memorización100posiciones:83→99aciertos;100posiciones nuevas78→79.250updates/149.6s con cerebro completo. DAgger256episodios:5x5victorias1/31→5/31;7x7y mayores0/32. No se cumple criterio para ampliar dificultad ni reanudar50k. Híbrido permanece pausado en14,384, checkpoint sin cambios confirmado porSHA256. Seguimiento PAUSED. Informe research/DIAGNOSTICO_APRENDIZAJE.md.
