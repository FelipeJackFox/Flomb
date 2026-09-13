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

## Comparación espacial completada

runs/spatial-diagnostic-001:2semillas×2entradas×200updates, cerebro completo desde ganancias cero y cabezas nuevas emparejadas.256posiciones entrenamiento,128reservadas,128tableros de juego; distribuciones de minas disjuntas. Codificador local3x3 FIJO, energía de entrada igualada, sin parámetros extra. Reservadas:actual66/128y59/128;local67/128y61/128. Victorias5x5 actual13/62y12/62;local10/62y11/62.7x7 actual0/64y1/64;local0/64en ambas. No mejora clara; no adoptar ni reanudar50k. Híbrido sigue pausado en14,384, SHA256 comprobado intacto. Informe research/RESULTADO_ENTRADA_ESPACIAL.md.

## Piloto de capacidad entrenable completado

`runs/capacity-diagnostic-001`: tres arquitecturas, dos semillas y 120 actualizaciones por modelo, minibatches emparejados de 16. 512 posiciones de entrenamiento, 126 reservadas y 48 tableros autónomos separados. Muestreo equilibrado por tamaño y deducción elemental/relacional. CNN: 98 y 99 aciertos reservados; arquitectura actual: 63 y 60; conectoma expresivo: 74 y 64. Victorias 5×5: CNN 10/24 y 8/24, actual 7/24 y 5/24, expresivo 7/24 y 4/24. En 7×7: CNN 0/24 y 1/24; ambos conectomas 0/24 en ambas semillas. Maestro visible: 18/24 y 21/24 respectivamente.

El expresivo conserva el grafo completo y aprende encoder convolucional, dos canales por neurona, ganancias, sesgos, mezcla y retención. Tiempo de entrenamiento: actual 31–32 s; expresivo 143–145 s; CNN 0.70–0.74 s, excluyendo carga y evaluación. Es una comparación supervisada de arquitectura, no reproducción del optimizador QR original ni comparación igualada en parámetros/costo. El expresivo memoriza 466/512 y 468/512, pero no aporta mejora consistente en partidas nuevas; no adoptarlo todavía ni lanzar corrida larga. La CNN demuestra mejor generalización de decisiones en este piloto, pero tampoco resuelve 7×7 de forma fiable. Próximo paso recomendado: ampliar demostraciones y cobertura de trayectorias, y hacer ablaciones de entrada/lectura antes de más capacidad o RL largo.

Seis pruebas nuevas de gradientes/paridad/restauración/separación de datos y siete existentes pasan. PyTorch opcional en `experiments/requirements-capacity.txt`; scripts `capacity_probe.py`, `capacity_references.py` y `report_capacity_probe.py`. Checkpoints `.pt` del piloto no son compatibles con el híbrido. SHA256 del híbrido confirmado intacto al terminar; permanece pausado en 14,384. Resultados y límites: `research/RESULTADO_CAPACIDAD_ENTRENABLE.md`; protocolo: `research/PROTOCOLO_CAPACIDAD_ENTRENABLE.md`.

## Escala de datos y ablaciones autorizadas

`runs/scaled-learning-001/dataset.pkl`: 10,000 posiciones distintas de entrenamiento, 1,000 de validación y 500 partidas finales. 5×5/3 y 7×7/7 a partes iguales. Entrenamiento usa 1,555 layouts y validación 146: varias posiciones por tablero, sin compartir layouts ni estados visibles entre esos splits. Las 500 partidas se reservaron antes de recoger trayectorias y no aparecen en los datasets de los dos pilotos anteriores. Muestreo equilibrado por tamaño y deducción elemental/relacional. Se mantienen las aperturas centrales seguras originales.

Control CNN terminado, `runs/scaled-learning-001/cnn`: dos semillas, 3,000 actualizaciones, batch 64. Aciertos reservados 972/1000 y 970/1000. Victorias 5×5 excluyendo seis automáticas: 178/244 y 173/244; 7×7: 108/250 y 111/250. Aproximadamente 71 s de entrenamiento por semilla, sin evaluaciones. Es evidencia de aprendizaje autónomo bajo esta receta ampliada; no aísla datos frente a más actualizaciones ni prueba una ventaja del conectoma.

Comparación en curso: `runs/scaled-learning-001/ablations`, PID inicial 3864 (verificar siempre `launch.json` y comando), log `benchmarks/scaled-ablations.log`. Cinco modelos: CNN, actual, solo encoder, solo dinámica interna y ambos. 3,000 actualizaciones, batch 16, semilla 20260924; mismos minibatches. Las cuatro variantes del conectoma conservan un canal, cabeza inicial idéntica y todo el grafo. Es un análisis inicial con una semilla, no una decisión de adopción. La CNN emparejada ya terminó: 977/1000, 160/244 en 5×5 y 111/250 en 7×7; 17.8 s de entrenamiento.

Guardado atómico cada 100 updates con pesos/Adam/RNG; `--resume` valida configuración, fuente y hashes. Evaluación de partidas por lotes validada frente a serial. 16 pruebas pasan, incluyendo restauración del CLI con pesos y Adam exactamente iguales. `caffeinate` acompaña el PID durante esta comparación. Informe regenerable: `experiments/report_scaled_training.py` → `research/RESULTADO_ESCALA_Y_ABLACIONES.md`. Protocolo: `research/PROTOCOLO_ESCALA_Y_ABLACIONES.md`.

Híbrido original sigue pausado en 14,384. Automatización existente `avisar-al-terminar-buscaminas` actualizada para vigilar exclusivamente esta comparación cada 30 minutos, en silencio durante avance normal; avisará al finalizar/fallar, actualizará informe/estado y se eliminará al dejar de ser necesaria. No reiniciar automáticamente ni cambiar presupuestos. Los resultados CNN ya se comunicaron; falta notificar la comparación del conectoma cuando termine.

## Comparación terminada y notificada · 2026-09-13 05:44 UTC

Verificados completed.json, cinco resultados de 3,000 actualizaciones y los cinco checkpoints `.pt` con pesos finitos y Adam; SHA256 del híbrido coincide con el manifiesto. PID 3864 ya no está activo, como corresponde a la finalización. Informe `research/RESULTADO_ESCALA_Y_ABLACIONES.md` regenerado y leído.

Comparación emparejada batch 16, una semilla: CNN ganó 160/244 en 5×5 y 111/250 en 7×7; actual 101/244 y 4/250; solo encoder 112/244 y 8/250; solo dinámica interna 91/244 y 8/250; ambos 103/244 y 15/250. Tiempos de entrenamiento respectivos: 17.8 s, 750.2 s, 859.8 s, 1129.6 s y 1274.3 s. Oportunidades seguras aprovechadas: 2506/2635, 1102/1420, 1413/1730, 1126/1499 y 1397/1743. Muertes con jugada segura disponible: 67, 206, 204, 239 y 215 respectivamente.

La CNN de esta tabla usa el mismo presupuesto que los conectomas; no confundir con las dos CNN iniciales de batch 64. Los componentes nuevos dan señales limitadas, pero ninguna variante del conectoma se acerca al control CNN. No adoptar ni declarar ventaja biológica con una semilla; no reanudar el híbrido ni lanzar nuevas pruebas automáticamente. Las 1,000 posiciones reservadas pertenecen a 146 tableros, no son 1,000 ensayos independientes. Finalización comunicada en este aviso; seguimiento eliminado al no quedar entrenamiento autorizado activo.

## Interfaz espacial y plasticidad por conexión · en curso

Usuario autorizó la prueba estructural. `runs/structural-learning-001`: conserva los 10,000 ejemplos de entrenamiento y 1,000 reservados, pero usa 500 partidas finales nuevas, sin compartir layouts con los conjuntos previos ampliado/espacial/capacidad. La nueva interfaz usa coordenadas de columnas ópticas anotadas: 4,426 neuronas L1/L2/L3 reciben entrada; diez tipos Mi/Tm/T1/C3 se interpolan por casilla y comparten decoder. Son decisiones de ingeniería, no visión validada. No hay entrada directa de pistas al decoder; solo actividad neuronal y contexto constante por tablero.

Plasticidad real por conexión: 25,582,938 multiplicadores independientes entrenables, signos preservados, sobre los 166,700 nodos retenidos. También se entrenan ganancias por neurona. Se reportan las conexiones efectivamente modificadas: tener un parámetro no implica gradiente en todas las conexiones. Control reconfigurado con grados de entrada/salida, pesos entrantes, signos y autoconexiones preservados; 13,310,849 swaps aceptados y 63.75% de asignaciones fuente-peso cambiadas. No preserva grado ponderado de salida ni representa una aleatorización completa.

Kernel C++ verificado para gradientes por conexión sin matrices densas N×N o E×batch. Benchmark real: batch 16 ~0.90 s/update y ~0.056 s/ejemplo, frente a ~0.111 s/ejemplo con batch 8. Pico acumulado del proceso de preparación ~2.69 GB. 21 pruebas pasan, incluyendo derivadas contra diferencias finitas y cálculo denso, independencia de plasticidad, invariantes del grafo y restauración del CLI con pesos/Adam exactamente iguales.

`runs/structural-learning-001/training`: CNN, retina_fixed (conexiones fijas, ganancias neuronales entrenables), retina_plastic y rewired_plastic. 3,000 updates, batch 16, semilla 20260926 y mismos minibatches. PID inicial 49087, verificar launch.json y comando; log `benchmarks/structural-training.log`. CNN ya terminó: 970/1000 posiciones, 159/228 victorias 5×5 excluyendo 22 aperturas automáticas, 109/250 en 7×7; 18.8 s de entrenamiento. Retina_fixed verificada avanzando en update 100, ~83 s. Checkpoints atómicos cada 250 updates, progreso cada 100. `caffeinate` acompaña al PID.

Cada variante neuronal se evaluará además con actividad anulada y permutada. La anulación implica scores constantes por construcción: su caída verifica dependencia funcional, no utilidad de la anatomía. El grafo reconfigurado entrenado es el control pertinente para esa cuestión. No adoptar con una semilla ni lanzar DAgger automáticamente. Híbrido sigue pausado en 14,384 y hash verificado intacto.

Automatización `avisar-prueba-estructural-de-buscaminas` activa cada 30 minutos, silenciosa durante avance normal; verifica checkpoints/intervenciones e integridad al terminar, regenera `research/RESULTADO_INTERFAZ_Y_PLASTICIDAD.md` con `experiments.report_structural`, notifica y se elimina cuando ya no sea necesaria. Protocolo en `research/PROTOCOLO_INTERFAZ_Y_PLASTICIDAD.md`. Estimación inicial de la comparación completa: 2–3 horas, no garantía de duración.

## Prueba estructural terminada y notificada · 13 septiembre 2026

Al responder a «como va», verificados completed.json, cuatro resultados, cuatro checkpoints de 3,000 actualizaciones con pesos finitos/Adam y las seis intervenciones zero/shuffle. El PID 49087 terminó; hash del híbrido intacto. Informe `research/RESULTADO_INTERFAZ_Y_PLASTICIDAD.md` regenerado y leído. Seguimiento `avisar-prueba-estructural-de-buscaminas` eliminado; no quedan entrenamientos de esta prueba activos.

CNN: 159/228 victorias 5×5, 109/250 en 7×7, 0.31 min de entrenamiento. Retina con conexiones fijas: 73/228 y 9/250, 38.47 min. Retina plástica: 68/228 y 11/250, 42.28 min. Grafo reconfigurado plástico: 90/228 y 17/250, 46.21 min. No hay mejora clara por habilitar plasticidad sobre el cableado original; el control reconfigurado obtuvo resultados nominalmente superiores, sin demostrar superioridad general con una semilla.

Oportunidades seguras aprovechadas: CNN 2719/2870, retina_fixed 979/1315, retina_plastic 1098/1441, rewired_plastic 1355/1661. Se modificaron 2,106,242 ganancias de conexiones en retina_plastic y 8,682,974 en rewired_plastic; disponibilidad de todos los parámetros no implica actualización de todas las conexiones. Zero produce 5/228 y 0/250 en los tres modelos; shuffle reduce fuertemente su desempeño. Esto verifica dependencia funcional de actividad/interfaz, no ventaja biológica. No adoptar modelos ni lanzar nuevas corridas/DAgger automáticamente; híbrido original sigue pausado.

## Diagnóstico funcional de escape terminado · 13 septiembre 2026

Autorizado tras investigar los videos. `runs/escape-functional-001`: grafo FlyWire comunitario fijado a commit 26672e06427c12c61536ce1bd93dae7442944681, 139,255 neuronas y 3,732,460 conexiones; hashes y tipos de las poblaciones verificados. No sustituye MaleCNS ni altera checkpoints. Los originales FlyWire no se reconstruyeron independientemente.

LIF comunitario, estímulo de acercamiento, 300 ms: DNp01 44/39/42 disparos en semillas 0/1/2; cero al silenciar conjuntamente LC4/LPLC2, conservando idénticas entradas. Brian2 independiente sobre el mismo grafo y eventos (semilla 0): 37 disparos y cero con ambas poblaciones cortadas. Replicación funcional, NO paridad numérica: el motor comunitario no debe asumirse equivalente al orden temporal de Brian2.

Nuestra normalización/dinámica tanh, con ganancias unitarias y mismas poblaciones, también transmite señal: pico DNp01 0.01937/0.02041/0.02041 en 3/6/12 ciclos; cero con ambas entradas cortadas. Esta prueba NO respalda que la simplificación destruya ese circuito. Entrada tasa/150 y salida neuronal directa son adaptaciones de ingeniería, no el encoder/decoder entrenado de Buscaminas. Quieto/alejándose tienen cero entrada por construcción del encoder; no prueban selectividad aprendida.

Normalizar pesos del LIF sin recalibrar sus parámetros elimina disparos de salida: escalas de pesos no intercambiables, no prueba de que normalizar sea malo para tanh. Se recomienda localizar pérdida de información en tareas mínimas de Buscaminas antes de cambiar todo a LIF. No se lanzó entrenamiento nuevo ni se simuló Dinosaurio/cuerpo. Informe y figura: research/RESULTADO_CIRCUITO_ESCAPE.md, research/escape-functional.png. Verificaciones emparejadas e independientes pasan en experiments.report_escape; fuentes y hashes conservados en la corrida.
