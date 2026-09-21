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

## Habilidad elemental y lectura espacial · terminado 13 septiembre 2026

Usuario autorizó aislar una habilidad mínima y localizar información. `runs/elementary-probe-001`: 600/240/240 posiciones sintéticas con pistas correctas, 5×5/3 minas, tres clases equilibradas (seguro/mina/indeterminado desde una pista designada). Máscaras ocultas separadas 50/17/17; sin solapamiento de máscaras ni observaciones. Aperturas parciales controladas, no todas alcanzables con expansión automática de ceros. No es win rate ni evaluación de partidas.

Checkpoints retina_plastic/rewired_plastic congelados; lectores diagnósticos MLP nuevos, tres semillas, elegidos por validación. Original: entrada visible 99.17%, encoder 96.81%, entrada neuronal 96.39%, vecindad tras ciclo3 96.53%, solo casilla 74.17%, vecindad comprimida a 10 valores (PCA solo train) 94.44%, capa oculta decoder 74.03%, score 46.25%. Reconfigurado: vecindad ciclo3 96.11%, casilla 65.56%, PCA10 86.81%. CNN nueva sobre la regla 240/240 en tres semillas. No son tres semillas del conectoma, y 240 posiciones comparten 17 máscaras de prueba.

El acceso espacial al contexto es un candidato concreto de mejora; no se demuestra pérdida irreversible ni fallo por score escalar, que se entrenó para otra tarea. PCA10 y casilla10 igualan anchura/arquitectura del lector. Extracción comprobada contra forward original (errores <1e-6), etiquetas y splits verificados, híbrido intacto. Informe research/RESULTADO_HABILIDAD_ELEMENTAL.md y gráfica research/elementary-probe.png. Se recomienda probar un decoder espacial de actividad con campo 5×5, congelando inicialmente cerebro/encoder, luego evaluar partidas reales nuevas sin maestro. Esa intervención no fue lanzada.

## Decoder espacial congelado · terminado 13 septiembre 2026

Autorizado implementar y comparar lector espacial. `runs/spatial-decoder-001`: encoder, cerebro y mapa óptico congelados desde retina_plastic-20260926; lectores nuevos con ~12.8k parámetros: puntual 1×1/106 canales y espacial 3×3/32 canales dos capas (campo 5×5). Mismos 1,000 updates/batch64/lr0.001/semilla20261002, seleccionados por validación. Actividad de 10,000 posiciones reales y 1,000 de validación cacheada, sin bypass de pistas. Entrenamiento de cabezas ~un minuto total; preparación de caché separada.

Al reservar partidas se detectó universo finito: C(24,3)=2,024 layouts 5×5, de los cuales 1,897 ya se habían usado. Se conservó caché y se acotó búsqueda a los 127 inéditos restantes, más 250 7×7, sin consultar resultados. 377 partidas idénticas para original/puntual/espacial; sin maestro. Se excluyen 14 victorias por apertura automática 5×5.

Victorias: original 29/113 (25.7%) y 14/250 (5.6%); puntual nuevo 29/113 (25.7%) y 11/250 (4.4%); espacial 40/113 (35.4%) y 26/250 (10.4%). Comparación emparejada espacial vs original: gana solo espacial 12/18, solo original 1/6, por tamaño. Oportunidades seguras espacial 312/364 y 828/1044 frente a original 257/321 y 553/757, en trayectorias distintas. Es mejora piloto consistente con acceso espacial útil, no superioridad general ni solución del juego.

Tres pruebas del head pasan; extracción de caché contra forward original en ambos tamaños con error máximo0. Anulación de actividad del espacial: 3/113 y 0/250; verifica dependencia funcional, no ventaja anatómica. Checkpoints originales protegidos, híbrido SHA25681b09d60395123c0591d8837d2a4419825af51c4a93acb0dbebee31702095ea5. Resultados completos/checkpoints/cachés/fuentes en corrida; informe research/RESULTADO_DECODER_ESPACIAL.md y figura research/spatial-decoder-results.png. No se sustituyó el agente servido ni se publicaron cambios. Una semilla; 5×5 inédito agotado. Siguiente validación deberá reservar un benchmark fijo sin retocar según sus resultados y usar semillas/dificultades nuevas, especialmente 7×7.

## Consistencia del decoder espacial · terminado 13 septiembre 2026

Usuario autorizó comprobar repetibilidad antes de ampliar entrenamiento. `runs/spatial-consistency-001`: semillas de lector 20261002 (piloto reutilizado), 20261003 y 20261004 (nuevas), cada una puntual/espacial con presupuesto idéntico de 1,000 updates, batch64, lr0.001, selección por validación. Mismo único cerebro congelado, mismas cachés train10,000/valid1,000; no son tres cerebros entrenados de forma independiente.

Benchmark reservado antes de entrenar: 500 layouts7×7/7 minas inéditos, únicos, sin solapamiento con datasets/benchmarks previos. 0 aperturas automáticas. Original25/500 (5.0%); puntual20/500 en las tres semillas (4.0%); espacial38/500,38/500,39/500 (7.6%,7.6%,7.8%). Mejora en3/3; media espacial7.67%, +2.67 puntos vs original y +3.67 vs puntual. IC95% bootstrap emparejado por tableros, condicionado a estos tres lectores: [+0.67,+4.67] pp vs original, [+1.73,+5.67] vs puntual. Son500 tableros compartidos, no1,500 independientes. Nivel absoluto todavía bajo; no demuestra superioridad biológica frente a CNN ni generalización a otros tamaños.

Memoización de inferencia verificada contra forward original y reordenado, con observación/contexto como clave y sin compartir acciones.13,232 hits/5,437 misses (~70.9% reutilización), no equivalente automáticamente al mismo porcentaje de speedup. Prueba unitaria comprueba distinción de observación/contexto, duplicados y aislamiento frente a mutación. Hashes de cachés/dataset/checkpoints preservados al finalizar. Informe research/RESULTADO_CONSISTENCIA_ESPACIAL.md, figura research/spatial-consistency-results.png, fuentes/pesos/resultados por partida/verificaciones en corrida.

Conclusión: conservar el lector espacial como candidato para siguientes experimentos. No se sustituyó agente servido ni se lanzó ampliación posterior automáticamente. Mantener este benchmark cerrado a selección de hiperparámetros; para próximas evaluaciones reservar layouts nuevos7×7 y otras dificultades. El universo5×5 inédito sigue agotado.


## Ampliación del decoder espacial · terminado 13 septiembre 2026

Autorizado ampliar entrenamiento del lector con el cerebro congelado. `runs/spatial-extended-001`: tres semillas 20261002/20261003/20261004, mismo encoder/grafo/mapa, head espacial de 12,769 parámetros, cachés train 10,000 / valid 1,000. Máximo 5,000 updates totales, batch 64, Adam lr 0.001; selección solo por validación y parada por estancamiento. Checkpoints elegidos en 3,500 / 3,250 / 4,500; entrenamiento detenido en 5,000 / 4,750 / 5,000. Tiempo total de entrenamiento ~237 s, incluida reconstrucción de las primeras 1,000 actualizaciones; evaluación aparte.

Los antiguos heads carecían de Adam. Se reconstruyeron las primeras 1,000 actualizaciones y se exigió igualdad bit a bit con cada head antiguo antes de continuar: pasan las tres semillas. Nuevos mejores/últimos checkpoints guardan Adam y RNG; verificados pesos/estado finitos. Hashes originales preservados.

500 layouts inéditos 7×7/7 minas reservados antes del entrenamiento, sin solapamiento con datasets/benchmarks anteriores, compartidos entre los seis modelos. Sin maestro ni victorias automáticas. Comparación emparejada 1,000 updates → ampliado: 45→61, 46→68 y 45→65 victorias sobre 500. Media 9.07%→12.93%, +3.87 pp. IC95% bootstrap por tablero condicionado a estos tres lectores: [+1.73,+6.07] pp. Son 500 tableros, no 1,500 independientes; un único cerebro congelado, no tres entrenamientos cerebrales. Mejora consistente en las tres semillas, nivel absoluto todavía bajo; no demuestra ventaja anatómica ni generalización a otros tamaños.

Informe research/RESULTADO_ESPACIAL_AMPLIADO.md y figura research/spatial-extended-results.png, inspeccionada visualmente. Fuentes, hashes, checkpoints, historias y resultados por partida conservados. No se sustituyó agente servido ni se lanzó fase posterior. Híbrido original sigue preservado; benchmark final cerrado a selección posterior.

## DAgger espacial · piloto en curso

Autorizado tras ampliar el lector. `runs/spatial-dagger-001`: semilla 20261002 predeterminada, cerebro congelado, continúa best-20261002 del ampliado con Adam. Tres rondas de 300 partidas 7×7/7 minas y 750 updates; DAgger beta=0 mezcla 32 posiciones antiguas equilibradas y 32 acumuladas nuevas por minibatch. Maestro solo etiqueta acciones certificadas seguras, nunca ejecuta clics ni rescata partidas. Control con 2,250 updates adicionales exclusivamente de datos antiguos y baseline sin cambios. Selección mínima pérdida en validación antigua, incluyendo baseline.

500 layouts finales y 900 de colección reservados antes de entrenamiento, disjuntos de todos los datasets/benchmarks anteriores. Cinco pruebas de colector/memo/lector pasan; el colector coincide con trayectorias independientes incluso en errores. Log benchmarks/spatial-dagger.log. No sustituir agente servido; resultados de colección no equivalen a comparación emparejada. Protocolo research/PROTOCOLO_DAGGER_ESPACIAL.md. Al terminar verificar hashes, etiquetas públicas y replays de checkpoints, generar informe con experiments.report_spatial_dagger.


## DAgger espacial · terminado 13 septiembre 2026

Piloto completado en `runs/spatial-dagger-001`. Se recogieron 4,495 posiciones públicas con alguna jugada certificada segura en 900 partidas autónomas: 1,331 / 1,500 / 1,664 por ronda. Ningún clic ejecutado por el maestro. Tres rondas de 750 updates tanto para DAgger como control; seleccionados por validación update adicional 2,250 DAgger y 1,250 control. Cerebro congelado y una sola semilla de lector 20261002.

Benchmark final idéntico de 500 layouts nuevos 7×7/7 minas: baseline 53/500 (10.6%), control de más entrenamiento con datos antiguos 66/500 (13.2%), DAgger 80/500 (16.0%). Sin victorias automáticas. Diferencia emparejada DAgger−baseline +5.4 pp, IC95% bootstrap [+2.2,+8.8]; DAgger−control +2.8 pp, IC95% [−0.6,+6.2]. Mejora respecto al inicio, pero todavía NO evidencia concluyente de beneficio específico de DAgger frente a igual presupuesto; intervalo incluye cero y solo una semilla.

Oportunidades seguras aprovechadas baseline 1,800/2,238, control 1,843/2,268, DAgger 2,126/2,507. Minas conocidas elegidas 203/200/181; muertes con alternativa segura 268/260/241. Las trayectorias y denominadores difieren. Pérdida de validación DAgger 1.839→1.807. No ajustar victorias ficticias por 50/50; contadores separados disponibles en summary.json.

Cinco pruebas pasan. Verificación independiente de todas las 4,495 etiquetas públicas y separación de layouts, hashes originales intactos y cuatro pares de restauraciones de checkpoints con siguiente actualización exactamente igual. Pesos, Adam/RNG, datasets, fuentes/hashes e informes conservados. Informe research/RESULTADO_DAGGER_ESPACIAL.md, gráfica research/spatial-dagger-results.png inspeccionada. No se sustituyó agente servido ni se lanzó otra fase. Próximo paso recomendado: repetición con las otras dos semillas y benchmark reservado nuevo para confirmar ventaja frente al control de presupuesto.

## Consistencia de DAgger espacial · en curso

Autorizada repetición con otras dos semillas. `runs/spatial-dagger-consistency-001` coordina nuevas `runs/spatial-dagger-20261003` y `runs/spatial-dagger-20261004`; reutiliza checkpoints del piloto semilla 20261002 sin reentrenamiento. Mismo presupuesto de tres rondas × 300 partidas / 750 updates y mismos 900 layouts de colección, intencionalmente compartidos para controlar tableros. Benchmark nuevo de 500 layouts 7×7/7 minas, excluye datasets, juegos y colección previos; compartido por nueve modelos. Selección por validación, sin consultar benchmark. Cinco pruebas pasan. Log benchmarks/spatial-dagger-consistency.log, protocolo research/PROTOCOLO_CONSISTENCIA_DAGGER.md. No reemplazar agente servido. Reportar nuevas semillas separadas y bootstrap por tableros, no 1,500 ensayos independientes.


## Consistencia de DAgger espacial · terminado 13 septiembre 2026

`runs/spatial-dagger-consistency-001` completada; nuevas semillas 20261003/20261004 terminaron tres rondas y controles de igual presupuesto. Colección intencionalmente compartida de 900 layouts; posiciones propias 4,292 y 4,373, 8,665 etiquetas públicas verificadas. Misma arquitectura/cerebro congelados y selección por holdout original. Checkpoints adicionales elegidos control 1,000 en ambas; DAgger 1,750 en ambas. Primera semilla 20261002 reutiliza piloto intacto (control 1,250, DAgger 2,250).

Nueve modelos en mismos 500 layouts nuevos 7×7/7 minas, sin aperturas ganadoras automáticas ni maestro. Victorias inicial/control/DAgger: semilla reutilizada 64/66/85; nueva 20261003 70/57/89; nueva 20261004 55/55/88. Media 12.60%/11.87%/17.47%. DAgger mejor en las tres. Diferencia media DAgger−control +5.60 pp, IC95% por tablero [+3.60,+7.73]; DAgger−baseline +4.87, [+2.80,+7.07]. Solo nuevas: +6.50 pp vs control, [+4.00,+9.10], +5.20 vs baseline, [+2.80,+7.70]. Intervalos condicionados a estos lectores: 500 tableros compartidos, no 1,500 independientes, ni tres cerebros. Evidencia repetida a favor de DAgger en esta configuración; desempeño absoluto aún bajo.

Oportunidades seguras DAgger 2,045/2,413, 2,195/2,566 y 2,108/2,465; baseline 1,816/2,207, 1,828/2,245 y 1,689/2,082. Muertes con alternativa segura inicial→DAgger 256→234, 248→230, 250→217; minas conocidas elegidas 189→169, 172→157, 190→161. Trayectorias/denominadores distintos, sin atribuir diferencias al azar solamente.

Cinco pruebas pasan; auditoría independiente verifica 8,665 etiquetas, separación final de todos los datos/layouts anteriores (incluye colección sin etiquetas), colección idéntica entre semillas, hashes intactos y ocho pares de reanudación de checkpoints con siguiente actualización exacta. Informe research/RESULTADO_CONSISTENCIA_DAGGER.md y figura research/dagger-consistency-results.png inspeccionada; fuentes y hashes conservados. Procesos terminados. No se sustituyó agente servido ni se lanzó fase siguiente. Recomendación: ampliar experiencia DAgger manteniendo cerebro congelado, con evaluación nueva reservada y controles de presupuesto; ninguna nueva corrida autorizada aún.

## Ampliación de experiencia DAgger · en curso

Autorizada ampliar experiencia manteniendo cerebro congelado. `runs/spatial-dagger-expanded-001` coordina tres subcorridas `spatial-dagger-expanded-20261002/3/4`, continuadas desde mejores DAgger previos con Adam/RNG. Tres rondas adicionales de 600 partidas nuevas / 750 updates. Control conserva buffer DAgger heredado fijo (50% original + 50% heredado); ampliado agrega nuevas posiciones al buffer (50% original + 50% agregado). Baseline intacto. Selección mínimo holdout original incluyendo baseline.

Reserva previa: 500 layouts finales y 1,800 de colección nuevos, excluidos de todos los datasets/juegos/colección previos. Compartidos entre semillas con trayectorias propias. Log benchmarks/spatial-dagger-expanded.log, protocolo research/PROTOCOLO_DAGGER_EXPERIENCIA_AMPLIADA.md. Cinco pruebas antes de iniciar. Mantener checkpoints originales y agente servido. Verificar resultados antes de declarar mejora; una sola arquitectura/cerebro congelado, tres lectores.


## Ampliación de experiencia DAgger · terminado 13 septiembre 2026

`runs/spatial-dagger-expanded-001` y sus tres subcorridas completadas. Tres rondas × 600 partidas nuevas / 750 updates adicionales por semilla. Cerebro/encoder/mapa congelados; pesos, Adam y RNG heredados de mejores DAgger anteriores. Control continuó con mezcla original + buffer DAgger heredado fijo; ampliado agregó nuevas posiciones. Mismo presupuesto, selección holdout original incluyendo baseline.

Se recogieron 27,898 posiciones nuevas: 9,648 / 9,220 / 9,030. Checkpoints ampliados seleccionados en updates locales 2,000 / 1,500 / 1,250, control 1,750 / 1,750 / 750. Las semillas segunda/tercera se eligieron durante segunda ronda: datos nuevos disponibles para los modelos seleccionados 9,648 / 5,925 / 5,904, además de la experiencia heredada. La tercera ronda de estas semillas no contribuye a los pesos elegidos.

Mismos 500 layouts nuevos 7×7/7 minas, sin maestro ni victorias automáticas. Victorias anterior/control/ampliado: 81/92/102, 89/100/103, 86/88/93. Media 17.07%/18.67%/19.87%. Ampliado supera nominalmente ambos brazos en las tres semillas. Delta vs anterior +2.80 pp, IC95% condicional [+0.93,+4.67]; vs buffer fijo +1.20 pp, [−0.60,+3.00]. No evidencia clara de beneficio específico de más experiencia frente a igual presupuesto sobre experiencia existente. 500 tableros compartidos, no 1,500 independientes; un único cerebro.

Errores evitables persistentes: minas deducibles seleccionadas anterior→ampliado 170→176, 155→179, 180→184. Muertes con alternativa segura 249→245, 224→245, 243→241. Oportunidades seguras ampliado 2,591/2,958, 2,330/2,708, 2,391/2,770; trayectorias/denominadores diferentes. No presentar victorias adicionales como mejora uniforme de razonamiento ni convertir pérdidas 50/50 en triunfos.

Cinco pruebas pasan. Auditorías independientes por semilla verificaron 27,898 etiquetas públicas, filas heredadas, splits, doce pares de restauración exacta, continuidad de Adam y hashes intactos; muestras de caché con error máximo 0 (36 posiciones). Fuentes/hashes, actividad y etiquetas por ronda, datasets agregados y pesos/Adam/RNG conservados. Informe research/RESULTADO_DAGGER_EXPERIENCIA_AMPLIADA.md y gráfica research/dagger-expanded-results.png inspeccionada. Procesos terminados; ningún agente servido sustituido. Siguiente recomendación: diagnosticar clics en minas deducibles/errores con alternativa segura antes de otra ampliación de presupuesto. Ninguna fase posterior iniciada.


## Diagnóstico de minas deducibles · terminado 13 septiembre 2026

Autorizado diagnosticar errores antes de ampliar presupuesto. `runs/mine-choice-diagnostic-001`: evaluación de heads ampliados sobre cachés de entrenamiento disponibles/validación para tres semillas y replay exacto de 500 partidas de semilla predeterminada 20261002. Replay reproduce por partida wins, clicks, known_mine_choices y safe opportunities/choices del benchmark previo. No se modificaron checkpoints ni agente servido.

En replay: 3,860 clics, 2,958 con alternativa segura y 902 sin ella; 102 victorias, 245 muertes con alternativa segura. Los 176 clics en minas certificadas ocurrieron TODOS con alguna alternativa segura. Prueba suficiente: 45 de una sola pista, 122 combinando pistas locales, 8 con conteo global, 1 enumeración exacta. La clasificación usa cierre acotado y no afirma minimalidad matemática. Omitir estados sin solución segura es un hueco real del colector, pero no explica los 176 fallos observados en esta trayectoria.

Sobre posiciones disponibles para el checkpoint seleccionado (no implica cada fila efectivamente muestreada), minas escogidas: 683/14,143, 480/10,217, 502/10,277; bastaba una pista en 170/152/144. Rondas posteriores a selección excluidas; validación separada. Hay fallos de ajuste incluso en el conjunto disponible, no solo generalización. La pérdida uniforme sobre todas las acciones seguras sí penaliza indirectamente no-seguras, pero no distingue mina probada de incierta entre objetivos cero.

Prueba diagnóstica adicional: copia temporal del head de semilla 20261002 sobre 32 errores de una sola pista tomados del buffer heredado, misma actividad/capacidad/pérdida, Adam nuevo lr0.001, batch32, clipping5. 0/32→32/32 elecciones seguras en 50 updates, mantenido hasta500. Copia no guardada/servida como agente. Es ajuste in-sample de casos seleccionados, no generalización ni prueba completa de representación; muestra que estos32 fallos sí son corregibles sin cambiar cerebro. No se demuestra necesidad de cambiar la pérdida ni que la simplificación cerebral sea la causa.

Dos pruebas diagnósticas pasan (certificado por pista y simetría de pérdida entre no-seguras; tolerancia float32). Hashes intactos. Informe research/DIAGNOSTICO_MINAS_DEDUCIBLES.md y figura research/mine-choice-diagnostic.png inspeccionada; ejemplo real partida6200700011, pista4 con4 vecinos ocultos y clic en uno de ellos. Fuentes/hashes, ejemplos y contadores conservados. Benchmark previo ahora inspeccionado para diagnóstico; cualquier intervención exige nuevos tableros finales.

Siguiente propuesta: priorizar en replay posiciones de entrenamiento donde el lector escoge mina certificada pese a alternativa segura, mezcladas con experiencia ordinaria, manteniendo arquitectura/pérdida/cerebro y control de muestreo uniforme a igual presupuesto. Evaluar tarea auxiliar de riesgo después, separadamente. Priorización NO implementada ni corrida nueva iniciada.

## Priorización de errores de minas · en curso

Autorizado priorizar errores, manteniendo cerebro/encoder/arquitectura/pérdida. `runs/prioritized-mines-001` coordina tres semillas; parte de mejores heads ampliados con Adam/RNG. Mismos buffers disponibles para los checkpoints anteriores, ninguna nueva trayectoria. Control: 32 originales equilibradas +32 uniformes DAgger; prioridad: 32 originales +16 uniformes +16 errores actuales de mina certificada con acción segura disponible. Refresco cada250, sin corrección de importancia. 1,500 updates por brazo, validación original mínimo cada250 incluyendo baseline. Nuevos500 layouts finales7×7/7 minas compartidos por nueve modelos. Log benchmarks/prioritized-mines.log. Test del pool verifica máscara legal, selección de error y exclusión de aciertos/acciones inciertas. Protocolo research/PROTOCOLO_PRIORIZACION_MINAS.md. No sustituir agente servido ni declarar mejora sin evaluar partidas.


## Priorización de errores de minas · terminado 13 septiembre 2026

`runs/prioritized-mines-001` completada: tres semillas, 1,500 updates por brazo, mismos datos/arquitectura/pérdida/cerebro, Adam/RNG heredados. Control32 originales+32 uniformes DAgger; prioritario32 originales+16 uniformes+16 errores actuales de mina certificada, refresco250,24,000 draws prioritarios por semilla. Selección por mínimo holdout original cada250 incluyendo baseline.

Benchmark500 layouts nuevos7×7/7 minas; UNA victoria automática en apertura inicial para todos. Reporte ajustado sobre499 partidas con decisiones. Victorias anterior/uniforme/prioritario: 102/104/107;102/95/89;86/92/86. Los contadores crudos de consola incluían+1 victoria y n500. Medias ajustadas19.37%/19.44%/18.84%. Delta prioridad−uniforme −0.60 pp, IC95% condicional[−2.34,+1.14]; vs anterior −0.53,[−2.07,+1.00]. No mejora reproducible ni justificación para adoptar esta variante; tampoco prueba universal contra priorización.499 tableros compartidos, un cerebro.

Minas deducibles elegidas uniforme/prioritario:159/163,147/158,161/162. Muertes con alternativa segura217/224,215/226,224/228. No mejora específica de errores en partidas. Últimos checkpoints prioritarios sí redujeron errores del buffer frente a últimos uniformes:579/619,387/490,381/481, pero no siempre elecciones seguras totales y no fueron seleccionados. Prioritarios elegidos500/1000/0; uniformes750/250/1250. Tercera semilla prioritaria conserva baseline. Últimos checkpoints no se evaluaron en benchmark final ni se cambió selección tras conocer resultados.

Test de pool pasa; auditoría independiente34,637 filas de certificados públicos, pools iniciales exactos, separación de benchmark, continuidad de Adam y12 pares de restauración exacta; originales intactos. Informe research/RESULTADO_PRIORIZACION_MINAS.md y figura research/prioritized-mines-results.png inspeccionada. Datos, máscaras de minas, snapshots de errores, fuentes/hashes y checkpoints conservados. No se sustituyó agente servido ni inició fase posterior.

Siguiente propuesta: evaluar selección de checkpoints por desempeño autónomo en conjunto de validación separado y conservar prueba final nueva; posible desajuste pérdida/juego todavía es hipótesis, no causa demostrada de este resultado. Cambiar distribución o añadir objetivo auxiliar son otras hipótesis que requerirían pruebas separadas.

## Selección por partidas · en curso

Autorizado comparar selección por pérdida con desempeño autónomo. `runs/game-selection-001`: sin reentrenamiento, tres semillas × dos brazos, candidatos anterior/mejor por pérdida/último guardado (no todos los intermedios existen). 500 layouts nuevos de validación y otros500 finales, registrados también en dataset.pkl. Selección por mayor número de victorias autónomas, desempate menos updates y orden fijo; excluir aperturas ganadoras automáticas. Sellar selección de seis brazos antes de evaluar final. Final solo unión de elegidos/baselines; deduplicar pesos idénticos. Log benchmarks/game-selection.log, protocolo research/PROTOCOLO_SELECCION_POR_PARTIDAS.md. Test de selección verifica prioridad por victorias y desempates. Conservar originales, ningún agente servido sustituido.


## Selección por partidas · terminado 13 septiembre 2026

`runs/game-selection-001` completada sin reentrenamiento. Tres semillas × dos brazos con candidatos anterior/mejor por pérdida/último guardado;14 conjuntos de pesos únicos validados. Reserva previa500 layouts7×7/7 minas de validación y500 finales, disjuntos de todo lo previo y entre sí; ambos registrados en dataset.pkl. Cero aperturas ganadoras automáticas en ambos conjuntos.

Regla por juego: máxima victoria autónoma, desempate menos updates y orden fijo. Selección sellada antes de evaluar final. Eligió últimos uniforme/prioritario para20261002, baseline paraambos de20261003, mejoruniforme y últimoprioritario para20261004. Cambiaron5/6 elecciones respecto a pérdida. Solo unión de elegidos y baselines evaluada al final,11 pesos únicos; nada de retocar elecciones con el resultado final.

Resultados finales pérdida→partidas: uniforme103→105,97→88,90→90; prioritario112→107,92→88,82→89, todos sobre500. Dos mejoras,tres empeoramientos,un empate. Medias uniforme19.33%→18.87%, prioritario19.07%→18.93%; conjunto19.20%→18.90%. Delta agregado−0.30 pp, IC95% bootstrap condicional[−1.50,+0.87]; uniforme−0.47[−1.87,+0.93], prioritario−0.13[−2.00,+1.60].500 tableros compartidos, no3,000 independientes; dos brazos relacionados y un cerebro. No mejora consistente ni justificación para sustituir selección anterior. Prueba limitada a3 candidatos guardados y una muestra de validación; no demuestra optimalidad de pérdida ni inutilidad general de seleccionar por partidas.

Test de regla pasa. Auditoría independiente reconstruye1,000 layouts, comprueba splits/prior overlap0, elecciones recalculadas, conteos, deduplicación de pesos, sello intacto y originales sin cambios. Memo57,665 hits/31,285 misses; no convertir porcentaje en speedup. Informe research/RESULTADO_SELECCION_POR_PARTIDAS.md y figura research/game-selection-results.png inspeccionada; fuentes/hashes, resultados, manifest y selección conservados. Ningún checkpoint entrenado/cambiado ni agente servido sustituido. No fase posterior iniciada.

## Comparación controlada de entradas · en curso

Autorizada: `experiments/compare_input_representation.py`, protocolo `research/PROTOCOLO_COMPARACION_ENTRADAS.md`, salida `runs/input-representation-001`. Tres pares desde cero, mismo lector12,769 parámetros/inicialización/minibatches, 5,000 updates Adam, datos originales10,000/1,000. Pistas one-hot vs actividad congelada, mismo contexto público y pérdida. Sin nuevos datos DAgger. Selección holdout cada250 y prueba final500 layouts nuevos, sellada antes de evaluar. Cache reusada y comprobada; pesos originales preservados. Codificación espacial y equivalencia inferencia/caché de entrada directa verificadas. No cambiar agente servido.

## Comparación controlada de entradas · terminado 13 septiembre 2026

`runs/input-representation-001` completado: seis lectores desde cero, tres pares con inicialización/minibatches idénticos por hash,12,769 parámetros,5,000 updates por modelo, datos originales10,000/1,000, mismo contexto y pérdida. Cerebro/encoder congelados previamente entrenados. Sin nuevos datos DAgger. Selección holdout sellada antes de evaluar.

500 layouts nuevos7×7/7minas compartidos,0 aperturas ganadoras automáticas. Victorias pistas/cerebro por semilla20261002/3/4:109/59,115/44,112/52. Medias22.40%/10.33%; diferencia+12.07pp IC95% bootstrap condicional por tablero[+9.00,+15.13]. Tres semillas de lector,un solo cerebro,500 tableros compartidos. Pasos elegidos pistas3750/3500/3750; cerebro3500/3250/4500. Minas conocidas elegidas pistas135/148/137 frente a cerebro183/192/200; oportunidades seguras varían según trayectorias.

Interpretación: coste claro de la representación cerebral actual para este lector/presupuesto, incluyendo encoder,dinámica,escala,pooling; no separa componentes ni demuestra incapacidad del conectoma. Pistas directas también se quedan en22.4%, por lo que no atribuir todo el estancamiento exclusivamente al cerebro. Estos lectores no llevan refinamiento DAgger; no interpretar10.33% como regresión del modelo servido~20%. CNN antigua tenía otra arquitectura(ReLU y una tercera conv dilatada; campo receptivo9×9 frente a5×5 actual), por lo que43.6% antiguo no es comparación controlada con este22.4%.

Test codificación/inferencia pasa. Auditoría independiente reconstruye500 layouts,solapamiento previo0,recalcula selección,conteos,sello y hashes. Igualdad metadatos de caché y16 mapas recomputados contra checkpoint pasan. Informe `research/RESULTADO_COMPARACION_ENTRADAS.md`, figura `research/input-representation-results.png` inspeccionada, summary y verification guardados. Checkpoints mejores/últimos con Adam/RNG conservados. Ningún modelo anterior alterado ni agente servido sustituido. No fase posterior iniciada.

Siguiente propuesta: adaptar conjuntamente encoder de entrada y lector espacial manteniendo conectoma fijo, frente a control congelado; requiere nuevo protocolo/benchmark. Es hipótesis de mejora, no solución demostrada.

## Interfaz conjunta · en curso 14 septiembre2026

Autorizado piloto encoder+lector frente a control congelado, `experiments/train_joint_interface.py`, `runs/joint-interface-001`. Parte del lector cerebral3500 de semilla20261002; conserva Adam/RNG para ambos.750updates,batch64(4micro16 conjunto),lrhead.001/lrencoder.0001, conexiones y ganancias fijas. Selección holdout0/250/500/750, prueba500layouts nuevos baseline/control/conjunto sellada. Test autograd disperso frente a gradiente denso pasa. Pruebas grafo real y hashes dentro del runner. Protocolo `research/PROTOCOLO_INTERFAZ_CONJUNTA.md`. No agente servido cambiado.

## Interfaz conjunta · terminado 14 septiembre2026

`runs/joint-interface-001` completado: piloto semilla20261002,750updates por brazo,headAdam/RNG heredados desde3500,mismos minibatches64 por hash. Conjunto aprende encoder lr.0001 y lector lr.001; control sololector. Conexiones,ganancias,mapeo,pooling,dinámica fijos. Control eligiópaso0(loss1.8386083);conjunto750(loss1.8300947). Curva conjunto0/250/500/750:1.8386083/1.8482665/1.8313025/1.8300947. Tiempo conjunto~662segundos incluyendo validaciones,control~12s; no afirmar igual tiempo de cómputo.

Final500 nuevos7×7/7minas,0 aperturas ganadoras automáticas:baseline56/500,control56/500,conjunto74/500.11.2%→14.8%,+3.6pp,IC95% bootstrap pareado condicional[+1.0,+6.2].Una semilla/un cerebro; mejora piloto,pendiente consistencia.No comparar con DAgger~20% en otros tableros. Minas deducibles elegidas177→154; elecciones seguras1817/2224→1834/2205,denominadores por trayectorias distintos.

Encoder control cambioL2=0;encoder conjuntoL2=.35840355. Pesos mejores/últimos,Adam,RNG preservados. Test disperso vsdenso pasa;grafo real forward/acumulación equivalentes,gradiente encoderL1=32.964 no nulo;resto cerebro intacto,validación online/caché inicial igual,memo vsforward restaurado igual. Auditoría independiente reconstruye500layouts,solapamiento0,selección/conteos/hashes/sello correctos.Figura `research/joint-interface-results.png` inspeccionada.Informe `research/RESULTADO_INTERFAZ_CONJUNTA.md`. Ningún modelo servido cambiado ni fase posterior iniciada.

Siguiente propuesta: repetir mismo protocolo en las otras dos semillas y evaluar en benchmark nuevo compartido para comprobar consistencia,antes de ampliar presupuesto o sustituir agentes.

## Consistencia interfaz conjunta · en curso14 septiembre2026

Autorizada repetición750updates en semillas20261003/4, mismo protocolo del piloto y checkpoint/Adam/RNG propio. Coordinador `experiments/repeat_joint_interface.py`, salida `runs/joint-consistency-001`, hijos `runs/joint-interface-20261003` y `runs/joint-interface-20261004`. Piloto20261002 preservado y reevaluado.500layouts nuevoscompartidos,sello de todos los candidatos antes de pruebafinal. Principal dossemillas nuevas,secundario las tres. Memo por encoder,dedup solo pesosiguales. Protocolo `research/PROTOCOLO_CONSISTENCIA_INTERFAZ.md`. Sin agente servido cambiado.

Seguimiento de esta repetición: automatización heartbeat `resultado-consistencia-interfaz-mosca` activa cada10min en esta tarea; silencio mientras siga normal, genera/audita informe e inspecciona figura al terminar, avisa resultado o fallo y se pausa. Runner observado activo PID78764, primera semilla nueva paso75/750 sin errores. Informe/auditoría preparados en `experiments/report_joint_consistency.py`, pendientes de ejecutar al completar. Duración estimada25–30min, no resultado todavía.

## Consistencia interfaz conjunta · terminado14 septiembre2026

`runs/joint-consistency-001` completado. Dos pares nuevos20261003/4 con protocolo750updates sin cambios; piloto20261002 reutilizado sin reentrenar. Se sellaron los seis checkpoints antes del benchmark final500layouts nuevos7×7/7minas compartidos,0 aperturas ganadoras automáticas. Todos los controles eligieronpaso0; conjuntos750.

Principal DOS semillas nuevas: control10.4%→conjunto13.4%,+3.00pp,IC95% bootstrap pareado condicional[+1.00,+5.00]. Individuales03:53→73/500(+4pp);04:51→61/500(+2pp,IC individual[-.8,+4.8],inconcluso por sí solo). Secundario tressemillas:10.8%→14.0667%,+3.2667pp,IC[+1.6667,+4.8667];piloto reevaluado58→77/500. Ambas nuevas tienen signo positivo y agregado favorable, sin prueba universal; un cerebro y500tableros compartidos, no1500observaciones independientes.

Minas deducibles control→conjunto por semilla:180→147,164→153,161→147. La adaptación de interfaz apoya una mejora modesta en este protocolo; no establece ventaja anatómica, rendimiento suficiente ni comparación directa con DAgger~20% de otros benchmarks.

Informe `research/RESULTADO_CONSISTENCIA_INTERFAZ.md` generado y figura `research/joint-consistency-results.png` inspeccionada. Auditoría independiente pasa:500layouts reconstruidos,solapamiento0,selecciones recalculadas,minibatches pareados,duplicados de pesos/conteos/sello/hashes intactos;seis modelos únicos evaluados. Originales y cerebro fijo intactos; ningún modelo servido sustituido ni nuevos experimentos iniciados. Monitor `resultado-consistencia-interfaz-mosca` se pausa al comunicar cierre.

## Extensión interfaz750→3000 · en curso14 septiembre2026

Usuario autorizó continuar y reportar sin detenerse. Se amplían tres pares a3,000updates, reanudando últimos750 con pesos/Adam/RNG exactos y conservando mejoresprevios;2,250updates adicionales por brazo. `experiments/extend_joint_interface.py`, `runs/joint-extended-001`, hijos`runs/joint-extended-SEED`. Misma pérdida,dataset,lr,clipping,microbatches;conectoma fijo. Todos los conjuntos previos eligieron750, motivación para probar presupuesto mayor,no garantía. Principal ampliado−conjunto750,secundario ampliado−control ampliado,500nuevos compartidos,sello previo. Testgradientes pasa y restauración de estado comprobada dentro del runner. Protocolo `research/PROTOCOLO_EXTENSION_INTERFAZ.md`.

Seguimiento reactivado `resultado-consistencia-interfaz-mosca` cada10min bajo instrucción explícita de reportar y continuar. Silencio en revisiones sin cambios; al completar ejecutar/auditar `experiments.report_joint_extended`, inspeccionar figura, reportar y elegir siguiente experimento local acotado basado en evidencia, documentarlo y actualizar monitor. Sin cambios al agente servido ni cómputo externo. Primera semilla verificada reanudaciónexacta head/encoder/Adam/RNG desde750;conjunto observado775/3000. Duración estimada1.5–2horas. Reporte preparado, pendiente resultados.

Hito comunicado14sept18:46UTC: primer par20261002 terminó3000updates;conjunto eligió2750,pérdida1.81405 frente a1.83009 en750;control conserva0. Verificaciones de entrenamiento pasan. Segunda semilla20261003 iniciada. Aún sin evaluar partidas finales; mejora de pérdida no es mejora de winrate.

Hito comunicado14sept19:28UTC: segundo par20261003 completó3000;conjunto eligió2750,pérdida1.83473 vs1.83750 en750,control conserva0;verificaciones pasan. Tercera semilla20261004 observada1325/3000. Dos de tres entrenamientos conjuntos terminados;partidas finales aún pendientes.

## Extensión interfaz750→3000 · terminado14sept20:09UTC

Trespares completados y auditados;500layouts nuevos compartidos,sin aperturasautomáticas. Principal conjunto750→ampliado15.533%→16.667%,+1.133pp,IC95%condicional[-.6,+2.802]: ganancia adicional no concluyente. Secundario controlampliado11.6%→16.667%,+5.067pp,IC[+3,+7.133]. Porsemilla conjunto750→ampliado81→83,76→88,76→79/500; controles56,59,59. Mejores conjuntos2750/2750/2000,controles0. Un cerebro,500tableros compartidos. No evidencia para seguir ampliando presupuesto solo por pérdida.

Auditoríaindependiente pasa(reconstrucción500,solapamiento0,restauración/historial/selección/pares/duplicados/conteos/hashes/sello). Figura `research/joint-extended-results.png` inspeccionada;informe `research/RESULTADO_EXTENSION_INTERFAZ.md`. Originales intactos y agente servido sin cambio.

## DAgger sobre interfaz adaptada · en curso14sept2026

Siguienteexperimento autorizado continuo: `experiments/adapted_dagger.py`, `runs/adapted-dagger-001`. Piloto20261002 parte de conjuntoadaptado2750;encoder+cerebro congelados,lector aprende.3rondas300partidasautónomas+750updates,controlsolo originales vs mezcla50%original+50%DAgger;teacheracciones0. Cache adaptada recomputada,Adamlector/RNG heredados,máscaras solo legales.900colección/500test nuevos disjuntos,selecciónsellada. Test unpaso demuestra igualdadexacta al filtrar Adam para congelarencoder. Protocolo `research/PROTOCOLO_DAGGER_INTERFAZ_ADAPTADA.md`.

Seguimiento actualizado a `runs/adapted-dagger-001` / `benchmarks/adapted-dagger.log`. Al completar ejecutar/auditar `experiments.report_adapted_dagger`,inspeccionar figura y reportar;continuar con repetición si mejora prometedora o siguiente hipótesis justificada si no. Extensiónya comunicada,no repetir. Cache adaptada en construcción al inicio;ningún resultado del piloto todavía.

## DAgger interfaz adaptada · terminado14sept20:31UTC

Piloto20261002 completado:control/baseline72/500(14.4%),DAgger108/500(21.6%),+7.2pp IC95%pareado[+4,+10.4] frente ambos.0aperturasautomáticas.4788posicionesnuevas,900partidascolección,teacheracciones0;lector2250updates,seleccionó2250;control0. Minasdeducibles146→136,eleccionesseguras1943/2341→2337/2668. Una semilla/un cerebro,prometedor pendienteconsistencia. Auditoríareconstruye1400layouts,solapamiento0,selección/conteos/hash/sello/encodercongelado verificados. Figuraadapted-dagger-results.pnginspeccionada,reporteresearch/RESULTADO_DAGGER_INTERFAZ_ADAPTADA.md. Originales/agenteservido intactos.

## Consistencia DAgger adaptado · en curso14sept2026

Se repite mismoprotocolo en20261003/4; piloto02 preservado y reevaluado sobre500nuevos compartidos. Nueva colección900layoutscompartida por lasdosreplicas con trayectorias propias. Principal dosnuevas,secundario tressemillas. Runner `experiments/repeat_adapted_dagger.py`,coordinador `runs/adapted-dagger-consistency-001`,hijos`runs/adapted-dagger-20261003/4`. Sellar todosloscandidatosantesdetest. Protocolo `research/PROTOCOLO_CONSISTENCIA_DAGGER_ADAPTADO.md`.

Seguimiento actualizado a consistenciaDAggeradaptado;informe/auditoría preparados `experiments/report_adapted_dagger_consistency.py`,pendientes de resultados. Primera repetición03 iniciada,cache adaptada en construcción. Al terminar informar dosnuevas/principal y tres/secundario y continuar con siguienteexperimento justificado. No volver a informar piloto como resultado nuevo.

## Consistencia DAgger adaptado · terminado14sept21:04UTC

Dosnuevas control18.6%→DAgger23.3%,+4.7pp IC95%pareado[+2.2,+7.1];03:102→112/500(+2pp,IC[-1.2,+5.2] individualinconcluso),04:84→121/500(+7.4pp). Secundariotres19.2667%→24.0667%,+4.8pp IC[+2.8,+6.8];pilotoreevaluado103→128/500.500nuevoscompartidos,0auto,uncerebro. Controleseligieron0/0/1500;DAgger2250/1750/2250. Signopositivo ambasnuevas y agregadofavorable,no universalidad ni ventaja anatómica. Auditoría1400layoutsreconstruidos,solapamiento0,selección/conteos/encoder/hash/sello pasan;figuraadapted-dagger-consistency-results.pnginspeccionada;informe RESULTADO_CONSISTENCIA_DAGGER_ADAPTADO.md. Originales/servidointactos.

## Comparación interfaces antes de DAgger · en curso14sept2026

Siguientepaso autorizado: `experiments/compare_dagger_interfaces.py`, `runs/dagger-interface-comparison-001`. Tresnoadaptadosnuevos desdecontrolprevio conigualpresupuesto vsadaptadosDAggerpreservados;DAgger3x300partidas+750updates,misma colección900 decontraparte porsemilla,final500nuevo. Comparapipelines,NOencoderaislado;selección previa difiere pero presupuestosnominalesigualados. Principaladaptado−noadaptado,pareado3semillas/500/uncerebro. Protocolo `research/PROTOCOLO_COMPARACION_INTERFACES_DAGGER.md`.

Seguimiento actualizado a comparacióninterfacesDAgger;reportero/auditoría `experiments/report_dagger_interfaces.py` preparados. Primera semilla noadaptada02 iniciada. Esperar cierre/figura/reportar antes de decidir nuevaintervención;no repetir aviso consistencia anterior. Originales protegidos y seguimiento continuo local.

## Comparación interfaces DAgger · terminado14sept21:44UTC

Trespares:sinadaptar87/82/84→adaptados125/104/100 sobre500nuevoscompartidos. Media16.8667%→21.9333%,+5.0667pp IC95%pareado[+2.8667,+7.2].0auto. Colección900porparejaidéntica,DAggerpresupuestosiguales,etapapreviaigualpresupuestonominal;compara pipelinesincluyendoencoder/lector/optimizaciónprevia,noencoderaislado. Un cerebro500compartidos. Auditoríapasa(layoutsfinales/colección/selección/encoderfijo/teacheracciones0/hashes/sello/conteos). Figura dagger-interface-comparison-results.pnginspeccionada;informe RESULTADO_COMPARACION_INTERFACES_DAGGER.md. Originales/servido intactos.

## Lector ampliado RF5→RF9 · en curso14sept2026

Siguientefase autorizadacontinua `experiments/train_wide_reader.py`, `runs/wide-reader-001`. Trespares local/wide desdeDAggeradaptado. Únicocambioarquitectura dilation/padding segunda conv1→3; mismos12,769param,pesos/Adam/RNG/batches,1500updatesextras,50%original+50%DAggerheredadofijo;brain/encodercongelados. Campo9×9 vs5×5 verificado porgradiente,pruebapasa. Nuevo500finalcompartido,selloantesdeevaluarbaseline/local/wide. Protocolo `research/PROTOCOLO_LECTOR_AMPLIADO.md`. No asumirqueRF9ve todo7×7desdecadaesquina ni quecerebrocarezcarecurrencianolocal.

Seguimiento actualizado a `runs/wide-reader-001`;reportero `experiments/report_wide_reader.py` preparado. Primera semilla en cacheexperienciaheredada. Test de campo/parametrización pasa. Al cerrar auditar/inspeccionar figura/reportar y continuar según evidencia sin otrodale. No repetir cierre comparacióninterfacesprevio.

## Lector ampliado · terminado14sept22:13UTC

Local21.8667% vsamplio16%,delta−5.8667pp IC95%pareado[−8.4,−3.4667]. Frente inicial20.7333%,amplio−4.7333pp IC[−7.2667,−2.2667].500compartidos/uncerebro,0auto. Local113/115/100;amplio96/83/61;baseline108/108/95. Amplioseligieron1500,todospeores;local1250/750/1250. Esta intervención conpesos/Adamheredadosperjudica,no demuestra inutilidad decontextoamplio entrenadodeotro modo. No adoptar ampliado. Auditoría yfigura wide-reader-results.png verificados,informe RESULTADO_LECTOR_AMPLIADO.md;originales/servido intactos.

## Auxiliar seguridad · en curso14sept2026

Siguientepiloto `experiments/train_risk_auxiliary.py`, `runs/risk-auxiliary-001`. Desde local02 elegido1250,fijarencoder/cerebro,lectorpolítica vslectorpolítica+0.2auxBCEcertificadosmine/safe;inciertasmascaradas. Mismosdatos/draws/headAdam/RNG,1500updates,b64,32original+32DAggerheredado. Ramaaux33parámetros,noacciones,nofiltromaestro. SelecciónmismaCEpolíticaholdout,selloantes500nuevos. Testgradientesinciertas/signos yforwardpolítica pasa. Protocolo PROTOCOLO_AUXILIAR_RIESGO.md. Hipótesisdistinta depriorizaciónya fallida.

Seguimiento actualizado a auxiliarseguridad;reportero `experiments/report_risk_auxiliary.py` preparado. Entrenamiento acabó ypasóaevaluación al configurar seguimiento. Ramaauxseleccionópaso0 segúnCEpolítica;últimoauxloss.31014 y7816/8781 certificadoscorrectos esdiagnóstico delúltimo,no confundirconcheckpointseleccionado. Esperarpartidas/auditoríaantesdeconcluir.

## Auxiliar seguridad · terminado14sept22:29UTC

Baseline/control/auxseleccionado117/500(23.4%),idénticastrayectorias,0auto. Ambosseleccionaronpaso0porCEpolítica: no se adoptóningúnupdate. Delta0/IC[0,0] refleja pesosdepolíticaidénticos,no pruebageneraldeequivalencia. Últimoaux1500(lossaux.31014,7816/8781certificadoscorrectos) NOeselmodelo evaluado. Auditoríarecomputacertificadospúblicos,todoslayouts/draws/selección/hash/sello/conteospasan;figura risk-auxiliary-results.pnginspeccionada;informe RESULTADO_AUXILIAR_RIESGO.md. No gananciapormétodoseleccionado.

## Lectura directa riesgo · en curso14sept2026

Diagnósticosinentrenar `experiments/evaluate_risk_readout.py`,`runs/risk-readout-001`:fijarúltimoaux1500ypreservadobaseline antesde500testnuevos. Principal decidirargminlogitmineaprendido vsheadpolíticadelMISMOúltimocheckpoint;secundariovsbaseline. Sinfiltromaestro,noprobabilidadcalibrada,inciertasnoetiquetadasduranteaux. Testindependenciadecapafinalpolíticapasa. Protocolo `research/PROTOCOLO_LECTURA_RIESGO.md`. Originales/servidointactos.


## Lectura directa riesgo · cerrada14sept2026

Auditoría completa: en500layouts nuevos7×7/7,baseline123/500(24.6%),últimapolítica117/500(23.4%),últimoriesgo48/500(9.6%),0auto. Principal riesgo−política −13.8pp IC95%pareado[−17.6,−10];vsbaseline−15pp [−19,−11.2]. Riesgo empeora claramente este piloto;no adoptar. Misma representación/checkpoint último para dos salidas, no reentrenamiento; inciertas no supervisadas, no probabilidades calibradas. Auditoría hashes/layouts/conteos verificada;figura inspeccionada;informe research/RESULTADO_LECTURA_RIESGO.md.

## Transferencia por dificultad · en curso14sept2026

Usuario pidió reportar y seguir sin parar. Nueva evaluación acotada sin entrenamiento: experiments/evaluate_difficulty_transfer.py, runs/difficulty-transfer-001, benchmarks/difficulty-transfer.log. Fijos tres best-SEED-local.pt de wide-reader-001 (02/03/04);200layouts nuevos compartidos por tamaño7×7/7,9×9/12,12×12/22,16×16/38. Total800layouts/2400episodios. Métricas autónomas por tamaño y semilla, decisiones seguras/oportunidades, minas deducibles, derrotas con segura y 50%exacto separado. No ajustar victorias por suerte. Protocolo research/PROTOCOLO_TRANSFERENCIA_DIFICULTAD.md,reportero experiments/report_difficulty_transfer.py. Caché limitada a50partidas y reiniciada para cada encoder. Cerebro/encoder/pesos congelados y protegidos;ningún cambio al agente servido. Objetivo orientar el siguiente entrenamiento hacia fallos observados antes de repetir cambios arquitectónicos.


## Evaluación por dificultad001 · fallo de mapeo detectado14sept22:52UTC

Completó2400episodios, pero auditoría de código detectó que activity_map solo agrupaba tamaños5/7 y devolvía mapas cero para9/12/16. Estos1800episodios NOevalúan generalización del cerebro. Conteos/layouts/hashes correctos no bastaban para detectar fallo semántico. 7×7válido:57/40/54victorias de200porsemilla,media25.1667%,ICcondicional[20.3333,30.3333]. Mayores0nointerpretables como capacidad;no reportarlos como aprendizaje fallido. Reportero y figura marcan invalidez;historias originales preservadas.

## Repetición dificultad002 con mapeo extendido · en curso14sept2026

Corrección implementada en retina_policy.build_mapping(sizes) y spatial_decoder.activity_map: mismos cuatrovecinos/ponderación delmapeo anotado,tablas9/12/16nuevas;validación temprana tamaño y rechazo sinmapeo. Tablasnuevas buffersno persistentes tras cargarcheckpoint;tabla5/7exacta. Prueba congraforeal pasa:paridadexactaactividad5/7,salida finita/no vacía/sensibleaobservaciones9/12/16,paddingcero,rechazo mapasfaltantes. Primera prueba tenía dosfixtures con igualobservación;se corrigiófixture exigiendo observaciones distintas antesde medir sensibilidad. Evidencia benchmarks/extended-mapping-verification.json.

Repetición lanzada: experiments.evaluate_difficulty_transfer --out runs/difficulty-transfer-002 --expand-mapping;log benchmarks/difficulty-transfer-corrected.log.800layouts NUEVOS excluyendo001,3políticas/2400episodios,sin entrenar;igualmétricas/presupuesto. Protocolo research/PROTOCOLO_TRANSFERENCIA_DIFICULTAD_CORREGIDA.md. Reportero alfinal --out runs/difficulty-transfer-002;figura difficulty-transfer-corrected-results.png,informe RESULTADO_TRANSFERENCIA_DIFICULTAD-CORRECTED.md. Checkpoints y agenteservido intactos. Evaluación mide transferencia desdeentrenamiento5/7coninterfazampliada.


## Transferencia dificultad002 corregida · terminada14sept23:11UTC

Evaluación válida con mapeo extendido:7×7 52/41/53de200,media24.3333%;9×9 3/2/0de200,media0.8333%;12y16todos0/200.800layouts nuevos compartidos/3políticas/uncerebro,0auto. Aprovechamiento seguro agregado7=88.1%,9=65.4%,12=51.7%,16=31.6%. Derrotasconseguradisponible7=234/454,9=386/595,12=312/600,16=357/600. No afirmar que restantes sean inevitables;solverlimitado. ICbootstrapdegenerado0si0éxitos,no pruebatasa verdadera0. Auditoríamapeo/hash/layouts/conteos pasan yfigura inspeccionada;research/RESULTADO_TRANSFERENCIA_DIFICULTAD-CORRECTED.md. Histórico entrenamiento solo5/7;interfazmayoresampliada sinentrenar.

## DAgger9×9 · en curso14sept2026

Siguientehipótesis exposicióna dominio nuevo: experiments/train_nine_dagger.py,runs/nine-dagger-001,benchmarks/nine-dagger.log. Piloto02desdelocalpreservado,mapeocorregido;3rondas300partidas9×9/12autónomas+750updates,control64originales vsDAgger32originales5/7+32nuevas9acumuladas. Cerebro/encoderfijos;Adam/RNGheredados. Presupuesto2250porbrazo;FIJARultimo2250antesdetest,mejorholdoutantiguo solo diagnóstico. Test500nuevos9+250nuevos7pareados frentecontrol ybaseline,colección900disjunta. Fuente protocolo research/PROTOCOLO_DAGGER_9X9.md.

Recolector adapted_dagger.collect generalizado tamaño/minas/contexto/solver/índices segúnpartida;prueba conservaexactamente7histórico yrecalculaetiquetaspúblicas9/accioneslegales,65posicionestest. Registro benchmarks/generalized-collector-verification.json. Firma main acepta mapping_path explícito,tablasnuevas no persistentes postcarga yviejasexactas. Reportero experiments/report_nine_dagger.py,auditaetiquetaspúblicas/layouts/hash/endpoint/encoder. No usarmejoressegúnholdout para eltestprincipal,solo latest2250. Sin cambios agente servido.


## DAgger9×9 piloto · terminado14sept23:38UTC

2250fijo9×9 baseline4/500,control2/500,DAgger13/500. Principal+2.2pp IC95%[+1,+3.6];vsbaseline+1.8pp[+.6,+3.2]. Aprovechamientosegurocontrol1112/1766(63%)→2567/3063(83.8%).7×7 baseline53/250,control45/250,DAgger42/250;delta−1.2ppvscontrol IC[−5.6,+3.2],vsbaseline−4.4pp[−8.8,0]. Ganancia9piloto limitada y posibleolvido7;no adoptar comoreemplazoglobal. Auditoría1650layouts,0solapamiento,4533etiquetaspúblicas recalculadas,endpoint/encoder/hashes/sello pasan. Figurainspeccionada research/nine-dagger-results.png,informe RESULTADO_DAGGER_9X9.md. Maestroacciones0,uncerebro/unasemilla.

## Consistencia DAgger9×9 · en curso14sept2026

Se repite mismoprotocolo en03/04desdelectorlocalpropio;piloto02preservado reevaluadosin entrenar. Coordinador runs/nine-dagger-consistency-001,runner experiments/repeat_nine_dagger.py,log benchmarks/nine-dagger-consistency.log;hijos runs/nine-dagger-20261003/4.900colección9nueva compartida dosréplicascon trayectoriaspropias;test5009+2507nuevocompartidotres. Principaldosnuevas9dagger-control,secundariotres/vsbaseline/retención7. Last2250fijoantesdetest. Protocolo PROTOCOLO_CONSISTENCIA_DAGGER_9X9.md. Reportero report_nine_dagger_consistency.py,auditaetiquetas/hash/coleccióncompartida/exclusiónhistórica/endpoint. No asumir2250layoutsindependientes ni3cerebros.

train_nine_dagger.main parametrizado(out,seed,splits,train_only),mantiene defaultpiloto;reserve acepta out explícito para que coordinador NOexcluya accidentalmentelospilotosprevios delregistro. Fuente archivada porrun. Agente servido yoriginales intactos.


## Consistencia DAgger9×9 · terminada15sept00:24UTC

Nuevas03/04en9×9 control0/1vsDAgger7/15de500;principalmedia+2.1pp IC95%[+1.2,+3.1],vsbaseline0/0 +2.2pp[+1.3,+3.3]. Pilotoreevaluado14/500vscontrol0/baseline1;3semillas+2.333pp[+1.467,+3.333].7×7 nuevasbaseline65/59,control51/57,DAgger53/51de250;DAgger−baseline−4pp[−7.6,−.4],vscontrol−.8pp[−4,+2.4]inconcluso.02baseline55/control51/dagger49. Mejora9repetida perobajaabsoluta,retención7empeorafrenteparent;noatribuir toda pérdidaalDAggerporquecontroltambiéncontinúaactualizaciones. Auditoríaetiquetaspúblicas/1650layouts/hash/encoder/endpointpasa;figurainspeccionada,informe RESULTADO_CONSISTENCIA_DAGGER_9X9.md.

## Repaso7 durante continuación9 · en curso15sept2026

Piloto experiments/train_seven_rehearsal.py,runs/seven-rehearsal-001,log benchmarks/seven-rehearsal.log. Parte latest-dagger2250 delpiloto9,1500updatesmásfijos(3750total);control32original+32experiencia9fija vsrepaso16original+16DAgger7heredada+32misma9. Mismopesos/Adam/RNG/índicesgenerados;solo se sustituyen16originales,primeros16originalesy32nueveidénticos. Grafoencoderfijos,no nuevaspartidastrain.750testnuevos(2507/5009),principal7replay-control,reportarcoste9 ycomparaciónconpadre9. Protocolo PROTOCOLO_REPASO_7X7.md,reportero report_seven_rehearsal.py. Recomputar8muestrasen3cachesconigualdadexactaantesentrenar. Agente servido yoriginalesintactos.

Preflightrepaso: igualdadbitabitactividadfalló porredondeofloat32alrecomputarconbatch8frentecachévariable,máximo1.01e-6. Preservado runs/seven-rehearsal-preflight-001 sinupdates ylogpreflight. Reiniciofreshcon750testnuevos;verificaencoder ylabels/context/maskexactos,actividadrtol1e-5/atol2e-6conmáximosregistrados. Protocolocorregido,noperderestadoentrenamiento(nohabíaempezado).


## Repaso7 piloto · terminado15sept00:48UTC

7×7padre9=48/250,control46/250,repaso57/250;principal+4.4pp IC95%[+.4,+8.4],vsparent+3.6pp[−.8,+8].9×9padre15/500,control13/500,repaso11/500;−.4ppvscontrol[−2,+1.2],−.8vsparent[−2.4,+.8]. Señalfavorable7vscontrol,gananciavsparentyno-inferioridad9sinconfirmar. Auditoríacache/índices/hash/endpoint/encoder/750layouts pasan;figurainspeccionada,informe RESULTADO_REPASO_7X7.md. Una semilla/un cerebro.

## Consistencia repaso7 · en curso15sept2026

Dosnuevas03/04conigual1500adicionalesdesdesusparent9latest2250;datos/caches/DAgger7propiosporsemilla. Runner experiments/repeat_seven_rehearsal.py,coordinador runs/seven-rehearsal-consistency-001,log benchmarks/seven-rehearsal-consistency.log,hijos runs/seven-rehearsal-20261003/4. Test750propiosporréplica(2507+5009),disjuntosentre sí ydelhistórico;principalmedia7replay-control dedosnuevas,secundariocoste9/vsparent. Piloto02NOreutilizadoenprincipal. Bootstrapestratificadoporsemillaypareadoporl ayout:1500layoutstotales/uncerebro. Protocolo PROTOCOLO_CONSISTENCIA_REPASO_7X7.md. Parametrizados train/report mantienendefaultpiloto,reporteshijosdentroderuncorrespondiente para no pisar piloto. Reportero global report_seven_rehearsal_consistency.py. Originalesyagente servidointactos.


## Consistencia repaso7 · terminada15sept01:20UTC

Dosnuevas7:03parent49/control57/repaso64de250;04parent60/control50/repaso61. Principalrepaso-control+3.6pp IC95%[+.6,+6.6],vsparent+3.2pp[−.2,+6.6].9:03parent10/control8/repaso10de500;04parent14/control11/repaso13;vscontrol+.4pp[−.3,+1.1],vsparent−.1pp[−1.1,+.9]. Recuperación7vscontrolrepetida,coste9inconcluso(noformalnoinferioridad),rendimiento9aúnbajo. Auditoríasindividuales+1500layouts únicos/disjuntos/hash pasan,figurainspeccionada,informe RESULTADO_CONSISTENCIA_REPASO_7X7.md. Piloto02excluido.

## Adaptación interfaz9 · en curso15sept2026

Siguientepiloto02: experiments/train_nine_interface.py,runs/nine-interface-001,benchmarks/nine-interface.log. Desde seven-rehearsal-001/latest-replay.pt,750updatesfijos con16original+16DAgger7+32DAgger9;controlsololector vsconjuntolector+encoder. Mismospesos/Adamlector/RNG/índices,encoderAdam nuevo1e-4,head.001heredado,clip5porgruposeparado. Grafo/aristas/gainsneuronales fijos;online4microbatch16 paraencoder;controlcacheverificada. Último750antesdetest5009+2507nuevos;principal9joint-control,7retenciónsecundaria. Protocolo PROTOCOLO_INTERFAZ_9X9.md.

JointInterface.activity ahora usa tamañosconmapeo registrado yrechaza faltantes;antes solo5/7igualquelarutacachehistórica. Testdense/gradientesexistente pasa. Preflightrealantesentrenarcompruebaforwardmezcla5/7/9,gradienteacumuladovscompleto,gradienteencoder9positivo,cachesactuales. Reportero report_nine_interface.py preparado;no reutilizarcachevieja trasadaptarencoder. Agente servido/originales intactos.


## Adaptación interfaz9 · terminada15sept01:54UTC

9×9: baseline16/500,control17/500,joint19/500;principal+.4pp IC95%[−1,+1.8],vsbaseline+.6pp[−1,+2.2]. Sin ganancia9concluyente.7×7baseline61/250,control55/250,joint63/250. Encoderjointcambió ycontrolidéntico,grafofijo;auditoríapreflight/pares/hashes/layouts/endpointpasa;figurainspeccionada,informe RESULTADO_INTERFAZ_9X9.md. No adoptarcomoavanceconfirmado.

## Diagnóstico asistencia pública · en curso15sept2026

Antesdeentrenarmás,experiments/evaluate_assistance.py,runs/assistance-diagnostic-001,benchmarks/assistance-diagnostic.log. Checkpointlatestjoint750 fijocomoobjetodiagnóstico,no ganadorelegido.500nuevos9×9/12 compartidos cuatrobrazos:autónomo,corregir50%propuestasqueomiten segura,corregir100%,solverpúblicocompleto(incluyeprobabilidadheurísticacuandoexactonoalcanza). Soloautónomoesrendimientodelagente. Sinentrenamiento,nicambiosagenteservido. Noatribuirvictoriasasistidasaaprendizaje niverloscomocotaóptima. Preflight compara16partidasautónomasconevalexistente,traceestadovisible/propuesta/acción/intervención en cadapaso para auditoríareconstruccióncompleta. Protocolo PROTOCOLO_DIAGNOSTICO_ASISTENCIA.md,reportero report_assistance.py. Lastrayectoriasdivergen,lasdiferenciasno soncomponentescausalesaditivosdelerror.


## Diagnóstico asistencia · terminado15sept02:24UTC

500nuevos9:autónomo19(3.8%),corrige50%44(8.8%),corrige100%275(55%),solvercompleto351(70.2%). Intervencionesdelegadas0/365/2048/9462;erroreseligibles424/723/2048/1518 respectivamente. SOLOprimeroautónomo;restoasistidos,nolearningni cotasóptimas. Auditoríareconstruyeestados/accionespúblicas/layouts/hashyparidadpasada,figurainspeccionada,informe RESULTADO_DIAGNOSTICO_ASISTENCIA.md. Trayectoriasdivergen,noparticióncausaladitiva.

## Cobertura con recolección corregida · en curso15sept2026

Hipótesis estadosavanzadosfaltanporqueautónomomuerepronto. experiments/train_deep_coverage.py,runs/deep-coverage-001,benchmarks/deep-coverage.log.400nuevoslayouts9colecciónpareadaautónoma vsseguridadcorregida100%;políticafijalatestjoint750,públicoseguros. Guardartraces/etiquetasprofundidad/ocultas;maestroactúasoloenrecolecciónexperimental.1500updatesporbrazo,mismosheadAdam/RNG,16original+16old7+16old9+16datosnuevosbrazo;primeros48idénticos,nuevosuniformescompartidosdistintostamañosdepósitos. Encoder/grafofijos actualadaptado9;RECOMPUTARtodasfeatures(no cachesviejas).

EvaluaciónfinalSIEMPREAUTÓNOMA750nuevos5009+2507frenteparent/controlautonomous-trained/assisted-trained,último1500prefijado. Protocolo PROTOCOLO_COBERTURA_PROFUNDA.md,reportero report_deep_coverage.py auditaacciones/etiquetaspúblicascolecciónytestseparado. Noatribuircambiossoloa profundidad porque cantidadestadostambiéncambia. Agenteservido/originalesintactos.


## Cobertura profunda · terminada15sept03:01UTC

Evaluaciónautónoma9:parent20/500,datosautónomos28/500,datoscorregidos22/500;principalcorregidos−autónomos−1.2pp IC95%[−3.2,+.8],vsparent+.4pp[−1.6,+2.4].7parent59/250,autónomos59,corregidos60. No mejora atribuibleacoleccióncorregida. Recogió5268posicionesvs2571,profundidadmedia12.9vs8.29,menosocultas30.46vs36.02;teacheracciones1883soloenrecolecciónexperimental. Máscoberturanobastóparaestelector/presupuesto. Auditoría pública/layouts/hashes/pares/endpointpasa,figurainspeccionada,informe RESULTADO_COBERTURA_PROFUNDA.md. Una semilla/un cerebro; no usarvictoriascolecciónasistidacomorendimientofinal.

## Masa segura · en curso15sept2026

Experimento experiments/train_safe_mass.py,runs/safe-mass-001,benchmarks/safe-mass.log. Parentdeep-coverage/latest-autonomous1500 (prefijadoparahipótesis),1500adicionaleslector,encoder/grafofijos. Mismosdatos16original+16old7+16old9+16new9autónomos,Adam/RNG/índices/batchesidénticos. SoloobjetivoCEuniformesobretodasetiquetassegurasvs−logmasaprobabilísticatotaldelconjuntoseguro;se permiteconcentrarenunasegura. Sinauxiliares/filtros/nuevacolección. Último1500fijopretest750nuevos5009+2507,principal9mass-uniform,retención7secundaria. Protocolo PROTOCOLO_MASA_SEGURA.md,reportero report_safe_mass.py.

4tests pasan:singletonCEvalor/gradiente,ilegalgrad0,invarianciarepartodentroseguras,desplazamientologits/todasseguras,rechazoetiquetasinválidas. Featuresrecomputadasconencoderactual. No afirmaréxitoporlossmenor(esobjetivodistinto). Agenteservido/originalesintactos.


## Masa segura · terminada15sept03:28UTC

9parent27/500,uniforme28/500,masa24/500;principal−.8pp IC95%[−2.8,+1.2],vsparent−.6pp[−2.6,+1.4]. No mejora9concluyente.7parent52/250,uniforme63,masa68. Auditoríapares/encoderfijo/hash/endpoint/layouts pasa;figurainspeccionada,informe RESULTADO_MASA_SEGURA.md. No adoptarcambioporguancia7puntualni lossdistinto.

## Comparación entradas con datos9 · en curso15sept2026

Nuevo diagnóstico experiments/compare_nine_representation.py,runs/nine-representation-001,benchmarks/nine-representation.log. Mismolector12769param NUEVOenambos,seed20261103,Adam.001nuevo,3000updatesfijos ybatchesidénticos16original+16old7+16old9+16new9autónomos. Raw10canalespistasvsbrain10actividadconencoderactualadaptado9fijo;amboscontextopúblico. Noheredarheadbrainpararaw. Test5009+2507nuevoscompartidos,principal9raw-brain,secundario7. Noigualacomputohistóricototal;una inicialización/un cerebro,noinviabilidadbiológica. Protocolo PROTOCOLO_REPRESENTACION_9X9.md,reportero report_nine_representation.py. Antigua comparacióninputs era7condatos5/7;esta prueba usa dominio9yencoderadaptado9. Rawescontrol,sinatribuirlecerebro. Agente servido/originalesintactos.


## Comparación representación9 · terminada15sept03:51UTC

Mismoslectoresnuevos/datos/3000updates:9raw62/500(12.4%)vsbrain18/500(3.6%),principal+8.8pp IC95%[+5.6,+12.2].7raw76/250vsbrain62/250. Brechaderepresentaciónbajoesteprotocolo,noimposibilidadbiológica,norawagenteconmosca. Encoder/grafoactual9preentrenados,igualpresupuestodellectornohistóricototal. Auditoría750layouts/hash/inicialización/batches/inputs/parametrización/endpoint pasa,figurainspeccionada,informe RESULTADO_REPRESENTACION_9X9.md.

## Tiempo de propagación · en curso15sept2026

Hipótesis:lectura temprana conserva información útil. experiments/compare_propagation_time.py,runs/propagation-time-001,benchmarks/propagation-time.log. Amboslectoresusanactividadgrafo;solo1cicloversus3antesdeagrupar10clasesdesalida. Mismosencoder/grafo/gain/mappingfijos;encoderhistóricoadaptadoa3(asimetríaexplícita). Lectoresnuevosseed20261103/12769param/3000updatesfijos,CEuniforme/Adam.001/mismosbatches16por4fuentes5/7/9. Cachesseparadasyciclorestablecidosegúncandidatoantesdeevaluar.750testnuevos5009/2507,principal9early-late. Protocolo PROTOCOLO_TIEMPO_PROPAGACION.md,reportero report_propagation_time.py. Ciclosnuméricosno equivalentesaHz/tiempobiológico. Agente servido/originalesintactos.


## Propagación piloto · terminada15sept04:23UTC

9early1=27/500vslate3=14/500,principal+2.6pp IC95%[+.2,+5].7early59/250vslate58/250. Señalfavorableuna inicialización,noadoptaraún. Encoderpreentrenadocon3fijo,mismosheadnuevos/Adam/batches/3000updates. Auditoríahashes/inputs/param/pares/endpoint/750layouts pasa;figurainspeccionada,informe RESULTADO_TIEMPO_PROPAGACION.md. Ciclos no tiempo biológico.

## Consistencia propagación · en curso15sept2026

Repetir conlectoresnuevos20261104/05,mismoencoder/grafofijo y3000updatesporbrazo. Runner experiments/repeat_propagation_time.py,coordinador runs/propagation-time-consistency-001,log benchmarks/propagation-time-consistency.log;hijos runs/propagation-time-20261104/5.750testpropiosporréplica5009/2507,1500únicos/disjuntos,principalmedia9early-late dedosnuevas,7secundario,pilotoexcluido. Protocolo PROTOCOLO_CONSISTENCIA_TIEMPO.md,reportero report_propagation_time_consistency.py,auditoríashijosylaidglobal. Sinpublicar nicambiaragenteservido;originalesintactos.


## Consistencia propagación · terminada15sept05:05UTC

Nuevas04/05en9early31/44vs late15/18de500,principal+4.2pp IC95%[+2.4,+6].7early70/57vs late62/50de250,+3pp[−1.4,+7.4].Ventaja9repetida,7inconcluso. Ambosgrafo/encoderfijospreentrenados3,lectoresnuevos3000updates.1500layoutstestúnicos;auditoríashijos/globalpasan,figurainspeccionada,informe RESULTADO_CONSISTENCIA_TIEMPO.md. Nociclosbiológicosni3cerebros.

## Adaptación interfaz temprana · en curso15sept2026

Siguientepiloto originalseed20261103 delprimerpar(prefijado,no escogermáximotest). experiments/train_early_interface.py,runs/early-interface-001,benchmarks/early-interface.log. Baselinecomponehead propagation-time-001/latest-early.pt yencoder deep-coverage/latest-autonomous,inrun baseline.pt sinmodificarfuentes. Ambos1ciclo,750updatesfijos,controlheadvsjointhead+encoder;Adamhead/RNGheredados,encoderAdam1e-4nuevo,head.001,clip5separado. Mismos16original+16old7+16old9+16new9. Grafo/gains/aristasfijos. TODAScaches1ciclo recomputadas yjointonline4microbatch16. Preflightrealgradientes/forward,mismo procedimientoauditado.750testnuevos5009/2507,principal9joint-control. Protocolo PROTOCOLO_INTERFAZ_TEMPRANA.md,reportero report_early_interface.py conauditoríaprovenanciabaselinecompuesto. Agenteservido/originalesintactos.


## Recuperación interfaz temprana15sept05:22UTC

Intento preflight archivado sin updates: diferencia cache lote16 vsdirecto4 de5.25e-6. Reproducción con mismo lote/deduplicación ahora debe ser exacta; diagnóstico4vs16 registrado, actividad entre lotes atol1e-5/rtol1e-5, otras verificaciones sin relajar. Reinicio en early-interface-001 con test nuevo, fuentes originales intactas.


## Interfaz temprana cerrada15sept05:46UTC

9baseline33/control36/joint44 de500; joint-control+1.6pp IC95[-.2,3.6].7baseline59/control69/joint66 de250,−1.2pp[-6,3.6]. Señal incierta. Auditoría pasa, figura inspeccionada. Siguiente consistencia early-interface-consistency-001,runner repeat_early_interface,log early-interface-consistency.log: padres tempranos04/05,750updates porbrazo,encoder común,1500test nuevos total. Protocolo CONSISTENCIA_INTERFAZ_TEMPRANA prefijado;reportero report_early_interface_consistency. Originales/agente servido intactos.


## Consistencia interfaz temprana cerrada15sept06:24UTC

9joint37/32vscontrol29/27de500;media+1.3pp IC95[-.2,2.7].7joint92/90vscontrol71/79de250;+6.4pp[3.4,9.4]. Mejora9 pequeña incierta;7 favorable nuevasréplicas. Auditorías hijos/global1500layouts pasan,figura inspeccionada. Siguiente runs/adapted-time-001,runner compare_adapted_time,log adapted-time.log:1vs2ciclos conencoderpilotoearly-interface/latest-joint fijo,headsnuevosseed06/3000updates,750testnuevos. Protocolo TIEMPO_ADAPTADO prefijado;reportero report_adapted_time. No promover agenteservido.


## Tiempo adaptado cerrado15sept06:49UTC

9uno44/dos38de500,+1.2pp[-1.6,4];7uno72/dos68de250,+1.6[-4,7.2]. Inconcluso. Auditoría pasa yfigura inspeccionada. Siguiente temporal-readout-001/compare_temporal_readout/log temporal-readout.log:concatenaciónactividad1+2vs1duplicada,15649params ambos,headsnuevosseed07,3000updatescadauno,750testnuevos. Protocolo LECTURA_TEMPORAL prefijado;report_temporal_readout. Preservar agente servido.


## Lectura temporal cerrada15sept07:20UTC

9combinado28/control38de500,−2pp[-4.4,.2];7combinado77/control81de250,−1.6[-8,4.4]. Sinmejora. Auditoría pasa,figura inspeccionada. Siguiente rotation-readout-001/evaluate_rotation_readout/log rotation-readout.log: mismo early-interface/latest-joint fijo1ciclo,identidad vs4rotaciones/logits inversos promedio,750testnuevos,0updates. Protocolo ROTACIONES;report_rotation_readout. Inferencia4x nominal,diagnóstico no aprendizaje.


## Rotaciones piloto cerrado15sept07:42UTC

9rotado119/500 vsidentidad39/500,+16pp IC95[12.4,19.8].7rotado98/250vs74/250,+9.6pp[3.2,16]. Diagnóstico favorable,sinupdates,4xinferencia nominal. Auditoría pesos/hash/750layouts pasa,figura inspeccionada. Siguiente rotation-consistency-001,repeat_rotation_readout,log rotation-consistency.log: modelosadaptados04/05,1500test nuevos disjuntos,principalmedia9sinpiloto. Protocolo CONSISTENCIA_ROTACIONES;report_rotation_consistency. Agente servido intacto.


## Consistencia rotaciones cerrada15sept08:14UTC

9rotado118/128vsidentidad39/38de500,+16.9pp[14.2,19.7];7rotado106/113vs69/69de250,+16.2pp[11.4,21]. Auditorías hijos/global1500layouts pasan,figura inspeccionada. Siguiente rotation-training-001/train_rotation_augmentation,log rotation-training.log:encoderpiloto fijo1ciclo,heads nuevosseed08/3000updates,originalvsrotaciónuniformeporposición/labels;ambos una inferencia.750testnuevos,protocolo ENTRENAMIENTO_ROTACIONES;report_rotation_training. Agenteservido intacto.


## Entrenamiento rotaciones cerrado15sept08:45UTC

9aumentado49/control43de500,+1.2pp[-1.6,4];7aumentado85/control73de250,+4.8[-1.2,10.8]. Incierto,una inferencia no replicaensemble. Auditoría pasa yfigura inspeccionada. Siguiente ensemble-distillation-001/train_ensemble_distillation,log ensemble-distillation.log:profesor fijo4orientaciones early-interfacepiloto,alumnos nuevosseed09/3000updates,CEvs.5CE+.5KL(T2),mismosdatos/encoder/grafo1fijos.750testnuevos,protocolo DESTILACION,report_ensemble_distillation. Agente servido intacto.


## Destilación cerrada15sept09:20UTC

9destilado32/control30de500,+.4pp[-2.2,3];7destilado90/control86de250,+1.6[-3.2,6.4]. Sinmejora clara. Auditoría pasa,figura inspeccionada. Siguiente reflection-readout-001/evaluate_reflection_readout,log reflection-readout.log:8simetrías vs4rotaciones,mismo modelopiloto early-interface/latest-joint fijo1ciclo,750testnuevos,0updates,2xcoste nominal. Protocolo REFLEJOS;report_reflection_readout. Agenteservido intacto.


## Reflejos cerrado18sept2026 (evaluación terminó15sept09:45UTC; reporte corrido18sept por Claude Code)

9ocho119/cuatro119 de500,+0.0pp IC95[-3.2,3.2];7ocho112/cuatro115 de250,−1.2pp[-6.4,4]. Sin mejora: los reflejos no añaden nada sobre las4rotaciones y cuestan2x. verification.json weights_fixed/originals_intact true,figura research/reflection-readout-results.png inspeccionada,informe research/RESULTADO_REFLEJOS.md. Línea de agregación en inferencia agotada: única ganancia robusta sigue siendo4rotaciones(+16.9pp en9,+16.2pp en7,replicada). Siguiente prueba SIN decidir ni arrancar;heartbeat codex sigue PAUSED. Agente servido/originales intactos. Visor temporalmente en8766(8765ocupado por asoiaf-map).


## Fusión de orientaciones · en curso 18 sept 2026 (Claude Code, autorizado por Felipe)

runs/orientation-fusion-001, runner experiments.train_orientation_fusion, log benchmarks/orientation-fusion.log, protocolo research/PROTOCOLO_FUSION_ORIENTACIONES.md, reportero experiments.report_orientation_fusion, tests experiments/test_orientation_fusion.py (5 pasan). Se descartó el lector equivariante de una sola pasada: la ganancia de las rotaciones viene de que el cerebro fijo no es equivariante, y eso solo se aprovecha con 4 pasadas. Tres brazos con 4 pasadas por jugada: mean (promedio de logits, control), fusion (lector sobre las 4 actividades alineadas), fusion_aug (con giros gratis desde caché). Lectores nuevos seed 20261110, 3000 updates, mismos draws; 750 layouts nuevos; principal 9×9 fusion_aug − mean. Encoder early-interface-001/latest-joint y grafo fijos, 1 ciclo. Agente servido/originales intactos.


## Aprendizaje sináptico · en curso 18 sept 2026 (Claude Code, autorizado por Felipe: "lo cool sería que aprenda el cerebro")

Conteo nuevo: con 1 ciclo solo 23,778 de 25,582,938 conexiones están en el camino entrada→salida (L1-L3→Mi/Tm, un salto); con 2 ciclos ~380k hacia salidas. La línea 1-ciclo usaba ~0.1% del conectoma. runs/synaptic-learning-001, runner experiments.train_synaptic_learning (arg = workers; reanuda solo si existe warmup.pt/latest-brain.pt), log benchmarks/synaptic-learning.log, protocolo research/PROTOCOLO_APRENDIZAJE_SINAPTICO.md, reportero experiments.report_synaptic_learning, tests experiments/test_synaptic_learning.py (3 pasan, incluye diferencias finitas float64). 3 ciclos; calentamiento lector nuevo 3000 updates (baseline); brazo reader sigue 3000 con cerebro fijo; brazo brain congela encoder+lector y entrena 25.6M log-ganancias por conexión + 166.7k por neurona 3000 updates con las mismas posiciones. Principal 9×9 brain − baseline, 750 layouts nuevos, una inferencia. PID 45157 con caffeinate, 5 workers, corre en paralelo con orientation-fusion-001 (PID 41320). Si hay señal, siguiente paso obligado: control con grafo recableado. Agente servido/originales intactos.


## Fusión de orientaciones cerrada 18 sept 2026

9×9: mean 109/500, fusion 118/500, fusion_aug 120/500; principal fusion_aug − mean +2.2 pp IC95 [−2.0, +6.4]; fusion − mean +1.8 [−2.4, +6.0]. 7×7: 90/91/89 de 250, ±0.4 pp [−6.8, +6.8]. Inconcluso: combinar las 4 lecturas con un lector aprendido no supera con claridad al promedio de logits al mismo coste. Auditoría del reportero pasa (hashes, sello, 750 layouts reconstruidos sin solape, mismos draws), figura research/orientation-fusion-results.png inspeccionada, informe research/RESULTADO_FUSION_ORIENTACIONES.md. No replicar: la línea de lectores queda cerrada en favor de aprendizaje sináptico (synaptic-learning-001, en curso). Agente servido intacto.


## Aprendizaje sináptico cerrado 18 sept 2026

3 ciclos, interfaz congelada, 3000 updates sobre ganancias por conexión/neurona (~4.1 s/update). Principal 9×9 brain − baseline: 26/500 vs 23/500, +0.6 pp IC95 [−1.6, +2.8]: nulo. 9×9 reader 24/500. 7×7: baseline 48/250, reader 72/250, brain 63/250; brain − baseline +6.0 pp [+0.8, +11.2], reader − baseline +9.6 [+4.8, +14.4], brain − reader −3.6 [−8.4, +0.8]. Holdout: baseline 1.796, reader 1.774, brain 1.765 (baja lento y casi plano desde update 500: 1.776→1.765). Conexiones modificadas 2,774,902 de 25,582,938 (755,868 con cambio >10%) y 63,135 neuronas: es EXACTAMENTE el conjunto que recibió gradiente en el preflight de 4 ejemplos, o sea el total de conexiones en caminos entrada→salida de ≤3 saltos con esta interfaz; el otro 89% del conectoma es inalcanzable por construcción. Lectura: las sinapsis solas sí aprenden algo (7×7 mejora sobre cerebro intacto, secundario, una semilla) pero no más que dejar aprender al lector de 12.8k parámetros, y 9×9 con 3 ciclos queda en ~5% (peor que la línea 1 ciclo: ~8% una inferencia, ~24% con 4 rotaciones). Auditoría del reportero pasa, verification.json ok, figura research/synaptic-learning-results.png inspeccionada, informe research/RESULTADO_APRENDIZAJE_SINAPTICO.md. No se corrió control recableado: sin señal en el principal no procede. Agente servido/originales intactos. Nada corriendo.


## Sonda de información cerrada 18 sept 2026 (diagnóstico, no benchmark de partidas)

runs/information-probe-001, experiments/probe_information.py, log benchmarks/information-probe.log. 9×9 de nine-dagger-001, 3000 posiciones train / 1000 test separadas por partida (150 partidas reservadas). Decodificadores por casilla reconstruyen la pista (tapada, 0..8) desde lo que ve el lector: lineal 1×1, lineal 3×3, MLP 3×3 (64 tanh). Acierto en casillas con número (mayoría global 45%): tablero crudo 100/100/100%; salida del encoder (3 canales) 73/89/98%; actividad tras ciclo1 55/72/83%, ciclo2 55/73/84%, ciclo3 50/65/81%, ciclo4 56/71/83%; con sinapsis aprendidas (synaptic-learning-001) 56/72/84% en ciclo1 y ≈ igual en el resto. Exactitud balanceada MLP: encoder 0.94 → ciclo1 0.61 (números altos casi perdidos). Lectura: la pérdida NO crece con los ciclos; ocurre entera en el primer salto entrada→salida (3 canales analógicos L1-L3 → sinapsis → pooling Mi/Tm). Ni el mejor lector local recupera ~17% de los números, y buscaminas exige números exactos: este es el cuello de botella de toda la línea, no el lector ni la profundidad. Las sinapsis entrenadas para jugar no restauran información. Siguiente propuesta (sin arrancar): entrenar la entrada (y luego sinapsis del primer salto) con objetivo auxiliar de reconstrucción en la salida, y/o ampliar tipos celulares de entrada. Agente servido/originales intactos. Nada corriendo.


## Entrada legible · en curso 18 sept 2026 (Claude Code, autorizado por Felipe)

runs/legible-entry-001, protocolo research/PROTOCOLO_ENTRADA_LEGIBLE.md. Etapa 1 experiments.train_legible_entry <encoder|synapses> <workers>, logs benchmarks/legible-entry-{encoder,synapses}.log (PIDs 64368/64369, caffeinate, ~0.5 y ~2.0 s/update, 3000 updates, objetivo solo reconstrucción de pistas tras el primer salto, 150 partidas de la sonda excluidas). Etapa 2 experiments.evaluate_legible_entry (sonda independiente + lectores nuevos + 750 partidas nuevas; log benchmarks/legible-entry-eval.log) se lanza sola al terminar ambos brazos. Principal 9×9 synapses − baseline. Agente servido/originales intactos.


## Entrada legible cerrada 18 sept 2026 · hipótesis refutada

runs/legible-entry-001 completo (etapa 1 ambos brazos 3000 updates; etapa 2 sonda + lectores nuevos + 750 layouts nuevos; aserciones internas de hashes/sellos/draws/cerebro congelado pasan; figura research/legible-entry-results.png inspeccionada; summary.json). Sonda independiente (acierto en números, MLP 3×3 / balanceada): baseline 83.1% / 0.61; encoder 93.3% / 0.89; synapses 95.6% / 0.84. Conexiones modificadas: 8,700. PERO victorias con lector nuevo de 3000 updates: 9×9 baseline 49/500, encoder 4/500, synapses 11/500; principal synapses − baseline −7.6 pp IC95 [−10.4, −5.0]; encoder − baseline −9.0 [−11.8, −6.4]; synapses − encoder +1.4 [0.0, +2.8]. 7×7: 69/32/40 de 250; −11.6 [−17.6, −5.6] y −14.8 [−20.8, −8.8]. Jugadas seguras aprovechadas 9×9: 0.894 / 0.811 / 0.853. Lectura: hacer legibles las pistas crudas EMPEORA el juego. El encoder original (entrenado para jugar) no transmite pistas: transmite rasgos ya procesados para la decisión, y al optimizar solo reconstrucción se destruyen. La sonda de pistas crudas medía lo que no importa; "83% legible" no era el cuello de botella. Implicación incómoda: una parte relevante del razonamiento ocurre en el encoder convolucional ANTES del cerebro. Dato a favor de las sinapsis: en ambas métricas de juego synapses ≥ encoder, con 8.7k conexiones. No usar entry-*.pt como entrada de juego. Nada corriendo. Agente servido/originales intactos.


## Nuevo enfoque: la mosca juega con recompensas biológicas · pilotos en curso 19 sept 2026 (Claude Code; Felipe ausente, objetivo: diagnosticar/corregir hasta mejora considerable)

Idea de Felipe: cuerpo de adorno, cerebro recibiendo estímulos que una mosca conoce, castigo/recompensa biológicos. Implementado sin backprop, sin encoder ni lector entrenados:
- experiments/prepare_mushroom_body.py → data/processed/mushroom_body.npz con conteos crudos de sinapsis del cuerpo pedunculado MaleCNS: 314 PN (88 glomérulos), 4,064 KC, 97 MBON, 316 PAM, 16 PPL1; PN→KC 22,586 conexiones, KC→MBON 61,210, PAM→MBON 2,693, PPL1→MBON 430.
- experiments/mushroom_body.py: sentidos → glomérulos → PN → KC (inhibición tipo APL, top-k) → MBON → valencia. Signo de cada MBON leído del cableado (más PPL1 = acercarse, más PAM = evitar; coincide con biología: MBON01-10 evitar, MBON11-14 acercarse). Solo cambian sinapsis KC→MBON con regla local de tres factores; mina = calor → PPL1, casilla segura = azúcar → PAM. Regla 'rpe' (dopamina señala sorpresa vía realimentación MBON→DAN) funciona; 'plain' casi no.
- experiments/fly_senses.py: (a) windows = vecindario 5×5 simbólico como bolsa de olores; (b) plume = física de arena: cada casilla abierta huele a "abierto" y una pista N reparte N unidades de olor "peligro" entre sus vecinas tapadas; la mosca huele 2 concentraciones.
- experiments/train_fly_player.py: la mosca elige casilla por softmax de valencia, pisa, aprende del resultado. Pilotos en benchmarks/fly-pilot/ (semillas dev 20M/21M, NO layouts reservados).
Hallazgos piloto: (1) sentidos simbólicos: la información llega a las KC (AUC lineal supervisado 0.77-0.84) pero la regla dopaminérgica extrae AUC ~0.70-0.74 y jugando gana 0-8/200 en 7×7. (2) plume: de 0 victorias (sin aprender) a ~10/200 en 9×9 y ~33/200 en 7×7 con 10-30k partidas; la valencia aprendida reproduce la heurística danger/abiertas (Spearman 0.95, AUC 0.806 vs 0.803). Techo de esa heurística jugando: 54/300 en 9×9 y 61/200 en 7×7. (3) cableado barajado rinde igual que el real: el mérito es del circuito tipo cuerpo pedunculado + dopamina, no de este cableado concreto. (4) Fallo detectado: casillas sin olor daban valencia 0 fija → la mosca prefería minas seguras a casillas desconocidas; corrección en prueba: olor de fondo de la arena + decaimiento de la tasa de aprendizaje. Advertencia honesta: en plume la división N/k la hace la física de la arena, no el cerebro; el cerebro aprende qué concentraciones predicen calor/azúcar.


## La mosca que olfatea y marca · evaluación formal cerrada 19 sept 2026 · MEJORA CONSIDERABLE

Diagnóstico previo (pilotos dev): con sentidos de pluma agregada (peligro total, nº de abiertas) la mosca se quedaba en ~5% en 9×9; un aprendiz TABULAR con memoria perfecta sobre esos mismos sentidos daba 16-29/300, o sea el cerebro ya estaba en la cota de esos sentidos: el límite era la información, no el circuito. Olor de fondo: empeora (descartado). Valencia relativa a la mosca ingenua + término tónico de contexto para casillas sin olor: se conserva. Sintonía en banda de los glomérulos (tuning=0.35): se conserva. Cambio decisivo: (1) olfatear por separado a cada vecina abierta (valencia = suma de respuestas MBON por olfateo) y (2) reflejo innato de marca de estrés cuando UN olfateo predice seguridad <3%; la marca absorbe una unidad del peligro de las pistas vecinas (semántica de bandera). Disparar la marca por la SUMA de olfateos producía marcas falsas en cascada; por el peor olfateo individual, cero marcas falsas.
Protocolo research/PROTOCOLO_MOSCA_MARCADORA.md (prefijado), runs/fly-marker-001, experiments/evaluate_fly_marker.py, log benchmarks/fly-marker-eval.log, figura research/fly-marker-results.png. 10,000 partidas de entrenamiento por mosca, solo calor/azúcar, solo cambian KC→MBON (27,558 de 61,210 sinapsis cambiaron en real-0), sin backprop/encoder/lector. 750 layouts reservados nuevos, 0 solapes con entrenamiento.
9×9 (n=500): real 284/275/190 → media 49.9% IC95 [46.3, 53.5]; barajado 249/296/294 → 55.9% [51.9, 60.0]; real − barajado −6.0 pp [−7.6, −4.4] (arrastrado por la semilla real-2; la semilla también cambia la firma olfativa); ingenua 0/500; sin marcas 12/500 (2.4%); política a mano (techo de estos sentidos) 318/500 (63.6%). 7×7 (n=250): real 61.2% [56.0, 66.4]; barajado 63.3%; ingenua 0.4%; sin marcas 12.4%; techo 71.6%. Marcas falsas: 0 en todos los brazos. Pisadas seguras 0.95-0.98 vs 0.67 ingenua. Valencia aprendida por olfateo (×100): share 0 → +7.4, 0.125 → +0.5, 0.25 → −0.7, 0.5 → −1.9, 1.0 → −6.2.
Referencia: la línea con backprop daba 9.8% en 9×9 con una inferencia (~24% con 4 rotaciones). Lectura honesta: (a) el salto viene de estímulos físicos bien elegidos + marcas; la división N/k y la resta de marcas las hace la arena, el cerebro aprende qué olores predicen calor/azúcar y de ahí qué marcar y qué pisar; (b) el cableado real NO supera al barajado: el mérito es de la arquitectura tipo cuerpo pedunculado + dopamina, no de este conectoma concreto; (c) varianza entre semillas grande (190 a 284). Pendientes: más semillas separando semilla de firma olfativa de semilla de cableado; aprender el umbral de marca; etapa 2 (MBON→descendentes y cuerpo visible en el visor). Agente servido/originales intactos. Nada corriendo.


## Validación de la mosca marcadora cerrada 19 sept 2026 (runs/fly-marker-002) · unidad = semilla

Origen: luz verde de Felipe + auditoría independiente. Verificado sobre 001: corr(valencia en 0.125, victorias)=0.897 entre 6 moscas; real−barajado −6.0 pp, EE 6.7, t=−0.89 (no se detecta diferencia); unión de victorias de real-0/1/2 = 66%, intersección 31% — la unión es cota oráculo, NO un voto jugable.
Arreglos: barajado con generador propio (real-s y shuffled-s comparten firma y sintonía), experiments/fly_agents.py (reproduce real-0 de 001 exacto: 284/500, pesos idénticos), experiments/test_fly.py (6 pruebas pasan: física de olfateos/marcas, sentidos sin acceso a minas, control pareado, contexto-referencia). Protocolo research/PROTOCOLO_MOSCA_VALIDACION.md, 750 layouts reservados nuevos, 10 semillas por grupo.
9×9 (% por semilla): real 54.8 51.4 34.0 2.2 54.8 46.6 48.4 48.6 51.8 52.8 → 44.5, DE 16.0, IC95 t [33.1, 56.0]; barajado 47.2 2.0 53.6 3.6 48.6 3.6 51.0 53.2 53.4 36.6 → 35.3, DE 22.8 [19.0, 51.6]; logística sin cerebro 53.4 52.4 49.0 53.2 56.4 53.4 2.8 49.0 52.0 53.6 → 47.5, DE 15.9 [36.2, 58.9]. Real−barajado pareado +9.3 [−6.2, +24.7], t=1.35, p=0.21. Real−logística −3.0, p=0.68. 7×7: real 60.8 [48.5, 73.1], barajado 49.6, logística 60.1. Ingenua 0.0. Política a mano 57.8% (9×9) y 69.2% (7×7) en estos layouts. Ensambles JUGADOS 9×9: ens-real-3 46.8, ens-real-10 53.2, ens-shuffled-10 55.4; unión oráculo de real-0/1/2 61.8. Marcas falsas 0 en todo.
HALLAZGO NUEVO: resultado bimodal. 5 de 30 aprendices (real-3, shuffled-1/3/5, logistic-6) colapsan a 2–4%: NUNCA marcan; su seguridad predicha para peligro 1.0 quedó en 3.4–8.9%, justo encima del umbral innato de 3% (las sanas: 0.7–2.8%; el siguiente estímulo, 0.667, anda por ~45%). Causa probable: al aprender que peligro 1.0 es malo dejan de pisarlo, se quedan sin muestras y la tasa de aprendizaje decae. Pasa también en la logística → no es del cerebro, es del umbral fijo. Entre las que sí marcan: real 49.2 (DE 6.4, n=9), barajado 49.1 (DE 6.0, n=7), logística 52.5 (DE 2.3, n=9).
Hipótesis: H1 cableado real = barajado → se sostiene (no se detecta diferencia; entre sanas 49.2 vs 49.1). H2 la logística iguala a la mosca → se sostiene, y es algo mejor y más estable: el circuito no aporta nada medible sobre un aprendiz lineal con estos sentidos. H3 cruce pista débil/casilla sin olor explica la varianza → se sostiene SOLO entre moscas que marcan (r=0.859, n=14); con las 20 es r=−0.36 porque domina el colapso. H4 ensamble > política a mano → SE CAE: jugado da 53–55% vs 57.8%; el 66% era unión oráculo. El ensamble de 10 sí supera a la mosca media y elimina el riesgo de colapso. Corrección a 001: su IC [46.3, 53.5] y su media 49.9% eran optimistas; con 10 semillas la media honesta es 44.5% [33.1, 56.0].
En curso: piloto dev del punto 2 (benchmarks/fly-reference-pilot/): actual vs contexto-referencia vs olor lejano de densidad vs olor lejano sin tónico.


## Punto 2 (robustez entre semillas) cerrado 19 sept 2026 · NINGÚN arreglo sobrevivió los pilotos

Todo en semillas y tableros dev (300 de 9×9 + 200 de 7×7), nunca reservados. Scripts: experiments/pilot_fly_reference.py, experiments/pilot_fly_dread.py; salidas benchmarks/fly-reference-pilot/, fly-dread-pilot/, fly-punish-pilot/, fly-punish2-pilot/.
1. Referencia para casillas sin olor (5 semillas dev, 9×9 / 7×7 / marcas falsas): actual 53.6 / 60.5 / 0; tónico solo en casillas sin olor (propuesta de la auditoría) 49.4 / 57.8 / 11; olor lejano de densidad sin tónico 50.7 / 58.3 / 13; olor lejano con tónico 43.1 / 51.8 / 658 (una semilla colapsa). Ninguna mejora a la actual.
2. Umbral de marca 3% → 10% / 20% sobre las semillas que colapsaron en 002 + sanas: rescata logistic-6 (4.3 → 55.0) y shuffled-5 (4.7 → 49.7) pero hunde real-3 y shuffled-3 (cientos a miles de marcas falsas, 0–3%) y degrada sanas (semilla 101: 53.3 → 33.0 con 12 marcas falsas). A 20% tres moscas marcan todo el tablero (7,669 marcas falsas, 0%). La semilla de sentidos 3 colapsa igual con cableado real que barajado → ese fallo viene del código olfativo, visible gracias al control pareado.
3. Castigo asimétrico (calor enseña k× más rápido). Con k=3 o 6 aplicado también al tónico: TODAS las moscas marcan todo (0%), porque el tónico es un offset global y arrastra todos los olores bajo el 3%. Solo en sinapsis: k=2 rescata shuffled-1 (4.0 → 47.3) pero colapsa la sana 102 (→ 1.3) y degrada la 101 (→ 33.0); k=3 vuelve a marcar todo. La logística con k=3 sobre todo el error sí se rescata (→ 54.3); con k=3 solo en pesos colapsa (2.3); con k=6 sobre todo el error marca todo (0%).
Lectura: el reflejo de marca con umbral fijo es un sistema biestable, de filo de navaja. Por debajo, la mosca nunca marca (~3%); por encima marca de más y se mata (0%). Aprender que peligro 1.0 es malo hace que deje de pisarlo y se quede sin muestras, así que la predicción se estanca justo encima del umbral en ~1 de cada 6 aprendices. Ningún escalar (umbral, ganancia de castigo, referencia) lo arregla para todas las semillas a la vez. No se corrió confirmación formal porque ningún candidato supera a la configuración actual en dev.
Lo único que sí mejora y ya está medido en layouts reservados (002): ensamble jugado de 10 moscas → 53.2% (real) / 55.4% (barajado) en 9×9 frente a 44.5% de la mosca individual media, sin riesgo de colapso. Código nuevo queda con defaults neutros (punish_gain=1, context_reference=False, far_field=False): reproduce 001/002. Pruebas 6/6. Nada corriendo.
Siguiente idea sin probar: que marcar no dependa de una probabilidad absoluta sino de evidencia acumulada por olor (p. ej. nº de quemaduras sin ningún azúcar), o que la mosca re-muestree olores temidos de vez en cuando; ambas atacan la causa (falta de muestras), no el síntoma.


## Análisis del mundo de sentidos · 19 sept 2026 (solo documento; sin código ni corridas)

A pedido de Felipe: research/ANALISIS_SENTIDOS_MOSCA.md. Criterio sentido↔trampa vuelto operativo en 5 pruebas (estado oculto, física que conoce las reglas, política de una línea, lector lineal, modalidad animal) y escala 0–4. Veredicto sobre lo actual: N/k y la resta de marcas son trampa (nivel 3); el olor lejano de densidad también; el reflejo de marca es política nuestra (gris). Propuesta central: olfateos crudos (N, k, m) con física genérica, con controles logística aditiva / tabla de pares / barajado; se acepta que el número puede bajar. De nuestro lado: entrenamiento no letal + currículum por densidad contra el colapso. Descartadas por fraude: patas que prueban minas, gradientes térmicos, olor de incertidumbre, olor de casillas compartidas, peligro iterado, moldeo con solver, castigo a marcas falsas. Pendiente decisión de Felipe; nada corriendo.


## DECISIÓN DE FELIPE · 19 sept 2026 · número oficial y cómo citarlo

Número oficial que se reporta: **ensamble jugado de 10 moscas = 53.2% en 9×9 (cableado real; 55.4% barajado), runs/fly-marker-002, 500 tableros reservados.**
LÉASE ANTES DE CITAR: eso es un ENJAMBRE QUE VOTA (promedio de valencias de 10 cerebros entrenados por separado), NO una mosca. No cuenta como "la mosca mejoró". Una mosca individual gana 44.5% en promedio, IC95 [33.1, 56.0] con la semilla como unidad, y 1 de cada 6 no aprende a marcar y se queda en ~3%. Además, ese resultado usa sentidos que hoy clasificamos como TRAMPA (la arena divide N/k y resta las marcas; ver research/ANALISIS_SENTIDOS_MOSCA.md): una regresión logística sin cerebro con los mismos sentidos saca 47.5%. Quien cite "53%" sin estas tres frases lo está citando mal.
Plan aprobado por Felipe (orden): 1) olfateos crudos (N, k, m) con controles logística aditiva / tabla / barajado; 2) entrenamiento no letal + currículum por densidad; 3) sentidos crudos de densidad + densidad variable; 4) traza de elegibilidad → marcar aprendido; 5) novedad interna. Instrucción explícita: si con sentidos crudos la mosca no aprende a marcar, decirlo sin suavizar; "el salto fue 100% de la arena" es un resultado legítimo.


## Punto 1 cerrado · olfateos crudos (N, k, m) · 19 sept 2026 · runs/fly-raw-001 · EL SALTO NO ERA 100% DE LA ARENA

Sentidos experiments/fly_senses.sniffs_raw (física genérica: cada casilla emite por su propio estado; suma en 8 direcciones; la arena no divide ni resta; marcar no altera N ni k). Protocolo research/PROTOCOLO_OLFATEOS_CRUDOS.md prefijado (criterio: ≤5% = "era de la arena"). 750 layouts reservados nuevos, 10 semillas por brazo, unidad = semilla, 10,000 partidas, resto idéntico a 002. Pruebas 8/8.
Piloto dev previo (benchmarks/fly-raw-pilot, curvas cada 5k hasta 40k): mosca en meseta desde 5k, ~30% en 9×9 (20–40 según lectura), sin tendencia hasta 40k; tabla 17–25% con parpadeo de marcas; aditiva 5–20%.
FORMAL 9×9 (media, IC95 t por semilla): mosca cruda real 34.0% [28.1, 39.9] (por semilla 29.0 31.2 29.2 40.4 33.2 46.6 48.0 23.4 27.6 31.2); cruda barajada 31.1% [25.2, 37.0]; tabla de triples sin cerebro 11.5% [5.3, 17.6] (5 de 10 nunca marcan); logística aditiva 7.5% [5.6, 9.5]; mosca con sentidos de trampa en estos mismos tableros 50.4% [37.9, 62.9] (una colapsada en 3.0); ingenua 0.0; política a mano 62.0. 7×7: cruda real 52.6% [48.2, 56.9]; barajada 49.7%; tabla 27.1%; aditiva 24.5%; trampa 59.4%; a mano 71.2%.
Contrastes pareados por semilla, 9×9: cruda − tabla +22.5 pp [16.3, 28.7], t=8.2; cruda − aditiva +26.4 [20.6, 32.2], t=10.3; cruda − trampa −16.5 [−31.8, −1.2], p=0.038; real − barajado +2.9 [−4.6, 10.4], p=0.41 (no se detecta diferencia). Marcas falsas 0. NINGUNA mosca cruda colapsa (0 de 20 nunca marcan), contra 1 de 10 con trampa y 5 de 10 tablas.
Lectura: (1) la mosca aprende sola la relación entre N, k y m a partir de calor y azúcar; conserva ~2/3 del rendimiento que tenía con la arena haciéndole la aritmética (34.0 vs 50.4). Aproximadamente un tercio del salto era de la arena, no el 100%. (2) Por primera vez el circuito aporta algo medible: con sentidos crudos le gana por >20 pp a los dos aprendices sin cerebro. Con sentidos de trampa la logística empataba; o sea que la trampa era justo lo que volvía al cerebro prescindible. Mi predicción de que la tabla quedaría arriba falló: la tabla aprende ~200 triples por separado y se queda sin muestras; la mosca generaliza entre triples parecidos por el solape de sus Kenyon. (3) Sigue sin importar el cableado real frente al barajado: el mérito es de la arquitectura (expansión dispersa + dopamina), no de este conectoma. (4) Sorpresa en contra: el enjambre que vota NO transfiere: ens-raw-10 da 27.8% en 9×9, PEOR que la mosca individual media (34.0%). El "número oficial" por ensamble vale solo para los sentidos de trampa.
Nota de citación: el número legítimo de UNA mosca es 34.0% [28.1, 39.9] en 9×9 con sentidos crudos. El 53.2% oficial es enjambre + sentidos de trampa.


## Punto 2 cerrado · entrenamiento no letal sobre olfateos crudos · 20 sept 2026 · runs/fly-raw-002

Piloto dev (benchmarks/fly-training-pilot, 20k partidas, evaluación letal): letal ~23–25% en 9×9; no letal ~50–57% y subiendo; currículum por densidad sin ganancia sola ni sumada → DESCARTADO, no entró a la formal.
Protocolo research/PROTOCOLO_ENTRENAMIENTO_NO_LETAL.md prefijado. 750 layouts reservados nuevos, 10 semillas, 20,000 partidas, sentidos crudos (N,k,m), evaluación SIEMPRE letal, unidad = semilla.
9×9 (media, IC95 t): mosca no letal 45.6% [41.4, 49.8] (52.6 42.8 44.0 32.2 51.2 45.2 47.6 49.4 42.4 48.4); mosca letal con el mismo presupuesto 32.7% [28.4, 37.1]; TABLA sin cerebro no letal 53.7% [51.1, 56.3]; aditiva no letal 15.1% [12.9, 17.3]; ingenua 0.0; a mano 61.4; enjambre de 10 moscas no letales 48.4 (solo dato).
7×7: mosca no letal 58.5% [56.9, 60.1]; letal 51.2%; tabla 60.6% [58.7, 62.6]; aditiva 38.0%; a mano 72.8; enjambre 60.8.
Pareados por semilla, 9×9: no letal − letal +12.8 pp [7.3, 18.4], t=5.3, p=0.0005; mosca − tabla −8.1 [−14.2, −2.1], p=0.014; mosca − aditiva +30.5 [24.9, 36.0]. 7×7: no letal − letal +7.3 [5.0, 9.5]; mosca − tabla −2.2 [−5.5, 1.2], p=0.18.
Nadie colapsa (0 de 40 nunca marcan). Marcas falsas: mosca no letal 22 en 9×9 y 1 en 7×7 (sobre ~35,000 marcas); resto 0.
Lectura sin suavizar: (1) La hipótesis de la escasez de muestras se confirma: entrenar sin morir sube a la mosca cruda +12.8 pp y reduce su varianza entre semillas (DE 5.8; con trampa era 16–17). Una mosca individual con sentidos legítimos llega a 45.6%, prácticamente lo que daba con la arena haciéndole la aritmética (44.5–50.4%). (2) PERO la ventaja del cerebro del punto 1 NO sobrevive: con muestras de sobra, la tabla de triples sin cerebro le gana a la mosca por 8 pp en 9×9. Lo del punto 1 era eficiencia muestral (generalizar entre triples parecidos cuando hay pocos datos), no capacidad. Con datos abundantes esa misma generalización se vuelve un lastre: confunde triples vecinos (de ahí sus 22 marcas falsas; la tabla, 0). (3) Lo que sí queda en pie: hacen falta CONJUNCIONES. La logística aditiva se queda en 15% aun con muestras de sobra, como predije: no puede tener a la vez certeza en N=k y gradación en lo de junto. La mosca (conjunciones al azar) queda entre la aditiva y la tabla exacta, que era el orden que esperaba en el análisis de sentidos. (4) El cableado real no se probó contra barajado en esta corrida (en 001 no se distinguían).
Incidente operativo: el primer lanzamiento de esta corrida habría escrito en runs/fly-raw-001 (los workers de multiprocessing con spawn reimportan el módulo y no veían el OUT del lanzador). Detenido al minuto, antes de cualquier escritura; verificado que 001 no tiene archivos posteriores a su completed.json (antes y después de esta corrida). Arreglo: Pool con initializer que pasa OUT/GAMES/GROUPS y seguro que se niega a escribir en un run con completed.json.
Pendiente del plan aprobado: 3) sentidos crudos de densidad + densidad variable; 4) traza de elegibilidad → marcar aprendido; 5) novedad interna. Nada corriendo.


## Punto 3 cerrado · sentidos crudos de densidad + densidad variable · 20 sept 2026 · runs/fly-density-001

Protocolo research/PROTOCOLO_DENSIDAD.md prefijado. Base: olfateos crudos + entrenamiento no letal, 20,000 partidas, 10 semillas, unidad = semilla. Sentido nuevo experiments/fly_senses.sniffs_far: solo en casillas sin vecinas abiertas, tres cantidades crudas de toda la arena (M contador tal cual, C olor tapado total con resolución de Weber, P feromona propia total); no dispara el reflejo de marca. Variable = densidad uniforme 8–25% por partida. Evaluación letal en layouts reservados nuevos a 10/15/20% de minas (300 de 9×9 + 150 de 7×7 por densidad). Pruebas 9/9.
9×9, % victorias (IC95 t) a 10 / 15 / 20% y promedio:
- mosca lejano+variable 85.6 / 55.2 / 15.5 → 52.1 [50.2, 54.0]; mosca cercano+variable 83.0 / 55.3 / 15.3 → 51.2 [48.8, 53.6]; mosca cercano+fijo 83.4 / 47.9 / 9.5 → 46.9 [44.5, 49.4]
- tabla lejano+variable 89.2 / 56.3 / 15.5 → 53.6 [53.4, 53.9]; tabla cercano+variable 83.0 / 57.1 / 14.7 → 51.6; tabla cercano+fijo 85.0 / 53.5 / 10.8 → 49.8; política a mano 87.0 / 67.3 / 26.3 → 60.2.
PRINCIPAL (mosca lejano − cercano, ambos variable, 9×9, promedio de densidades): +0.9 pp [−2.1, +3.9], p=0.51 → NO se detecta que el sentido de densidad ayude a la mosca. En la tabla sí ayuda: +2.1 [0.7, 3.4], p=0.007, casi todo a 10% de minas (+6.2 [5.3, 7.1]). La mosca muestra la misma dirección a 10% (+2.6, p=0.14; en 7×7 +3.7, p=0.07) pero no alcanza.
SECUNDARIO que sí salió: variar la densidad en el entrenamiento ayuda a la mosca, y ayuda INCLUSO en el 15% de siempre: variable − fijo +7.4 pp a 15% [3.9, 10.9], p=0.001; +5.7 a 20% [0.5, 11.0]; −0.4 a 10%; promedio +4.2 [1.8, 6.7], p=0.003. Yo había predicho que costaría entre 0 y 3 pp a 15%: fallé, salió al revés. Lectura no comprobada: la densidad variable muestrea mejor los triples (N,k,m) raros y afloja el sobreajuste a un solo régimen. En la tabla el efecto es menor (+3.6 a 15%, +3.9 a 20%, −2.1 a 10%). En 7×7 los efectos van en la misma dirección pero no se distinguen de cero.
Mosca vs tabla: con densidad variable ya no se detecta diferencia en 9×9 (lejano: −1.5 [−3.4, +0.3], p=0.09; a 15%: −1.1, p=0.44). La ventaja de 8 pp de la tabla en fly-raw-002 se achicó porque la densidad variable le sirve más a la mosca. La mosca sigue teniendo marcas falsas (37–55 en 9×9 sobre ~100 mil marcas) y la tabla ninguna. Nadie colapsa.
Mejor mosca individual legítima hasta hoy, en el 15% estándar de 9×9: 55.2–55.3% (con entrenamiento variable), contra 45.6% en fly-raw-002 y 34.0% en fly-raw-001. OJO al comparar: son conjuntos de layouts distintos y aquí se excluyeron aperturas que ganan solas (no hubo ninguna en los otros); la comparación limpia es la interna de esta corrida: 47.9 → 55.3.
Techo: la política a mano da 67.3% a 15%. Quedan ~12 pp que ya no son de un solo punto: son adivinanza con densidad y deducciones de dos pistas.
Incidente de rendimiento: la primera pasada no terminó ninguna mosca con campo lejano en 2 h 15 min (el costo por decisión crecía con todos los olores conocidos, miles con el campo lejano). Detenida; RawFlyAgent ahora usa solo los olores presentes en el tablero; verificado que reproduce un brazo ya terminado (1350/1350 resultados iguales, 665 victorias); reanudada conservando los 27 brazos hechos. Los brazos de antes y después del cambio son numéricamente equivalentes.
Decisiones que salen de aquí: adoptar entrenamiento con densidad variable; NO adoptar el campo lejano para la mosca (no ayuda y cuesta cómputo). Pendiente del plan: 4) traza de elegibilidad → marcar aprendido; 5) novedad interna. Nada corriendo.


## Visualización "Arena" · 20 sept 2026

scene/dist/arena.html (servida por el visor: http://127.0.0.1:8766/arena.html; en 8765 si se usa scene/serve.py tal cual). Tablero 3D estilo buscaminas clásico (three.js del vendor + assets/flybody.glb) junto a un 2D clásico sincronizado y un panel de cerebro (Kenyon activas, votos de las 97 MBON, valencia, lámparas PAM/PPL1). La mosca camina, olfatea una por una a sus vecinas abiertas (etiqueta N·k·m), planta sus marcas de estrés como banderas (con halo de feromona), da el toque y la casilla se abre en cascada; azúcar o explosión. Esferas de olor con radio ∝ N. Conmutadores: esferas, valencia por casilla; pausa, paso, velocidad, selector de partida.
Datos REALES: scene/export_arena.py entrena una mosca legítima (olfateos crudos, no letal, densidad variable, 20k partidas, semilla 0; pesos en runs/fly-viz/) y exporta 12 partidas de desarrollo (7 ganadas, 5 perdidas; semillas 23,000,000+, no reservadas) a scene/dist/data/arena.json. Esa mosca gana 48% en 200 tableros dev 9×9. No toca index.html ni el agente servido.


## Punto 4 cerrado · marcar como acción aprendida · 20 sept 2026 · runs/fly-marking-001 · APRENDE EN LA DIRECCIÓN CORRECTA PERO SE QUEDA MUY CORTO

Mecanismo experiments/fly_agents.LearnedMarker: marca con probabilidad σ((θ − z)/τ), z = log-odds de seguridad del olfateo más temido; traza de elegibilidad (decae 0.9 por pisada) y dopamina de las pisadas siguientes (azúcar +1, calor −3, relativo a promedio móvil) ajustan el único parámetro θ. Nadie le dice si la marca fue correcta. Protocolo research/PROTOCOLO_MARCAR_APRENDIDO.md. Base: olfateos crudos, no letal, densidad variable, 20k partidas, 10 semillas, 750 layouts reservados nuevos, evaluación letal y determinista. Pruebas 12/12. (La corrida se interrumpió a 44/61 brazos y se reanudó conservando lo hecho.)
9×9 (media, IC95 t): aprendido 36.9% [31.7, 42.1]; reflejo fijo 3% 52.8% [49.6, 56.1]; θ congelado en −12 16.3% [11.6, 20.9]; simulado (sham) 29.9% [23.2, 36.6]; tabla aprendida 27.8% [17.8, 37.8]; tabla reflejo 54.3% [52.2, 56.5]; a mano 61.8. 7×7: aprendido 51.7; reflejo 61.4; θ=−12 34.9; sham 51.0; tabla aprendida 45.5; tabla reflejo 62.0.
θ: el reflejo de 3% equivale a θ = −3.48. Aprendido: de −8 sube en las 10 semillas a −6.14 de media (finales entre −5.3 y −6.8), curva todavía subiendo muy despacio. Sham: se queda en −7.98 de media. O sea: la señal de crédito retrasado es real y sistemática.
ERROR MÍO EN EL PROTOCOLO: escribí que la mosca "empieza sin marcar nunca" con θ0 = −8. Es falso: con entrenamiento no letal el cerebro predice seguridades tan extremas para N=k que z cae por debajo de −8 (y hasta de −12), así que el brazo sham marca 5.5 casillas por partida y el de θ=−12 marca 3.3. El contraste que fijé como principal (aprendido − "sin marcas" = +20.7 pp) está contaminado y NO mide aprender a marcar. El contraste limpio es aprendido − sham: +7.0 pp [−1.4, +15.4], p=0.09 en 9×9; +0.8 [−4.3, +5.8] en 7×7. No alcanza significancia.
Contra el umbral puesto a mano: aprendido − reflejo = −15.9 pp [−21.9, −9.9], p=0.0002 (9×9) y −9.6 (7×7). El umbral aprendido (≈0.2% de seguridad) es mucho más conservador que el 3% y por eso marca menos (6.1 vs 7.5 por partida) y gana menos. La tabla aprendida sale peor aún (−26.5 vs su reflejo; 2 de 10 nunca marcan).
Lo único a favor: cero marcas falsas con el umbral aprendido (el reflejo de 3% tiene 21 en 9×9).
Decisión: se conserva el reflejo de 3% como configuración de trabajo, declarado como pieza puesta a mano. El mecanismo de traza funciona como prueba de concepto (θ se mueve con crédito real y no con crédito falso) pero no sustituye al umbral. Hipótesis sin probar de por qué se estanca: gradiente de política muy ruidoso con un solo parámetro y recompensa casi siempre +1; tasa que decae; pocas decisiones con z cerca de θ.
Pendiente del plan: 5) novedad interna. Nada corriendo.


## Inicio justo (apertura garantizada) y tableros grandes · 20 sept 2026 · runs/fly-start-001

Observación de Felipe viendo el visor: el inicio con UNA casilla central segura es injusto. Medido: en 9×9/12 esa apertura destapa una sola casilla con número en el 73.9% de las partidas (mediana abierta = 1). Estándar moderno: primera casilla Y sus ocho vecinas sin minas → abre en cascada. Implementado Minesweeper(..., zero_start=True) (mediana 40 abiertas en 9×9, mínimo 9); el default NO cambió, corridas previas reproducibles. Pruebas 13/13. Protocolo research/PROTOCOLO_INICIO_Y_TAMANO.md.
Receta: olfateos crudos, no letal, densidad variable, reflejo 3%; 8 semillas por brazo; layouts reservados nuevos; evaluación letal.
% victorias (IC95 t) por entrenamiento old / zero / zero-big, y política a mano:
- 9×9 inicio viejo: 54.1 [50.5, 57.7] / 49.3 [41.7, 57.0] / 46.7 [40.2, 53.1] · a mano 64.2
- 9×9 con cascada: 71.0 [68.9, 73.2] / 68.9 [61.4, 76.5] / 67.8 [60.2, 75.3] · a mano 78.8
- 16×16/40 con cascada: 40.0 [33.8, 46.2] / 38.9 / 37.0 · a mano 62.0
- 30×30/140 con cascada: 16.2 [11.7, 20.8] / 15.8 / 14.6 · a mano 40.0
Otras métricas (entrenamiento old): tablero despejado antes de morir 95.1% (9 cascada), 89.3% (16), 82.2% (30); pisadas seguras 98.4% / 99.1% / 99.6% (a mano 98.8 / 99.4 / 99.7).
(1) EL INICIO INJUSTO COSTABA ~17 PUNTOS: la MISMA mosca gana +16.9 pp [14.7, 19.1] en 9×9 solo por evaluarla con cascada (54.1 → 71.0). Felipe tenía razón. Las pisadas a ciegas bajan de 10.4% a 5.3%.
(2) Entrenar con cascada NO ayuda: zero − old = −2.1 pp [−9.9, +5.7], p=0.54 evaluando con cascada, y −4.8 (p=0.18) con inicio viejo; además mete más varianza entre semillas. Lectura sin comprobar: el inicio viejo obliga a practicar más situaciones difíciles (frontera pequeña, adivinanza), y eso transfiere; con cascada ve menos de eso (400k pisadas de entrenamiento vs 488k).
(3) Entrenar en tableros grandes NO ayuda: zero-big − zero = −1.2 (9×9), −1.9 (16×16), −1.2 (30×30), ninguna distinguible de cero, con más pisadas de entrenamiento (588k). Esperado: los sentidos son locales; un tablero grande o infinito solo cambia la proporción de bordes y la duración.
(4) La política aprendida en 7×7/9×9 transfiere tal cual a 16×16 y 30×30: 99.1% y 99.6% de pisadas seguras. Las victorias caen (40%, 16%) porque hay que encadenar cientos de decisiones; la política a mano también cae (62%, 40%). El hueco contra la política a mano crece con el tamaño (8 → 22 → 24 pp): un 0.3–0.4% más de error por pisada se paga caro en partidas largas.
Decisiones: EVALUAR de aquí en adelante con inicio en cascada (estándar); ENTRENAR con el inicio viejo (rinde igual o mejor y con menos varianza). No entrenar en tableros grandes. Número actual de una mosca legítima con inicio justo: 71.0% [68.9, 73.2] en 9×9/12 (política a mano 78.8%).
Pendiente: la visualización (scene/dist/data/arena.json) sigue con partidas de inicio viejo; regenerar con cascada.


## Repo con historial + ronda de optimización · 20 sept 2026

GIT: el repo no tenía commits. Hallazgo: Codex había dejado 169 instantáneas reales del árbol (objetos sin rama, refs/codex/turn-diffs) del 12 al 15 sept. Historial reconstruido en master: 169 commits [Codex] con esas instantáneas (fecha = creación del objeto; título = última sección nueva de ESTADO.md) + 15 commits de la etapa Claude Code, uno por hito, donde cada archivo entra con la versión vigente a esa fecha (copias que cada corrida guardó en runs/*/source, que conservan su mtime; ESTADO.md recortado a la sección del hito). Verificado: en hito/01 mushroom_body.py no tiene wiring_seed ni sniffs_raw; en hito/03 sí; zero_start solo en HEAD. Etiquetas: codex/ultimo, hito/01-mosca-marcadora … hito/07-inicio-justo. Límite honesto: dentro de cada hito nuestro no hay granularidad fina, y un archivo editado varias veces entre dos corridas solo conserva la versión que quedó copiada. .gitignore nuevo: se versionan código, research/, ESTADO.md, resúmenes/manifiestos/source de cada run y data/processed/mushroom_body.*; NO pesos, datasets, .pt/.npy/.pkl ni data/raw. Identidad FelipeJackFox / felipaupz@gmail.com. Sin remoto, sin push.
OPTIMIZACIÓN (mismo resultado, más rápido): (1) sniffs_raw con rebanadas de un tablero acolchado en vez de np.roll — 4000 comparaciones, 0 diferencias; (2) claves de olfateo vectorizadas (código entero por triple) y patrón Kenyon construido solo para la casilla que se pisa, porque la valencia es lineal (valor de casilla = suma de valores de sus olfateos); (3) pesos KC→MBON en formato disperso por Kenyon (solo el 15% de los pares son sinapsis reales) y lectura de valencia como un vector u mantenido incrementalmente. Verificación contra el código anterior: entrenamiento no letal 3000/3000 partidas idénticas (max |Δw| 5e-6); tabla 2500/2500 idénticas; en letal 2529/3000 idénticas y luego divergen por caos numérico (una diferencia de 1e-6 cambia un sorteo). Velocidad: mosca 0.0304 → 0.0098 s/partida (×3.1), tabla ×3.5. Una corrida de 20k partidas no letales baja de ~35 min a ~11 min por proceso.


## Frente 2 cerrado · ¿dónde le gana un cerebro a una tabla? · 21 sept 2026 · runs/fly-vs-table-001

Protocolo research/PROTOCOLO_MOSCA_VS_TABLA.md. 8 semillas, 300 layouts 9×9/12 reservados nuevos con inicio en cascada, evaluación letal, unidad = semilla.
A. EFICIENCIA MUESTRAL (% victorias; mosca / tabla / diferencia pareada). Entrenamiento NO letal: 250 partidas 33.4 / 12.8 / +20.7 [0.9, 40.4]; 500: 44.8 / 16.0 / +28.7 [16.5, 40.9]; 1k: 47.1 / 30.8 / +16.2 (ns); 2k: 54.9 / 44.4 / +10.5 (ns); 4k: 69.8 / 50.7 / +19.2 [4.2, 34.2]; 8k: 71.9 / 60.8 / +11.2 (p=0.09); 16k: 68.4 / 70.1 / −1.7 [−9.4, 6.0]. La mosca llega a su meseta (~70%) en 4,000 partidas (~100 mil pisadas); la tabla necesita 16,000 (~390 mil): ≈4× más experiencia. Entrenamiento LETAL: la mosca va arriba en toda la curva: 500: 25.2 / 5.5; 4k: 45.3 / 8.2 (+37.1 [25.3, 49.0]); 16k: 44.9 / 26.3 (+18.5 [8.4, 28.7]).
   Lectura: la ventaja del circuito es eficiencia muestral, ≈4×, y es grande cuando la experiencia es escasa o cara (letal). Con experiencia ilimitada la tabla empata. Es la primera ventaja del cerebro medida con limpieza en este proyecto. (Sigue sin depender del cableado real: no se probó barajado aquí; en corridas previas no se distinguía.)
B. RUIDO SENSORIAL (concentraciones × exp(σ·ε)), no letal, 10k partidas: σ=0: mosca 71.1 / tabla-entera 71.6 / tabla-fina 71.6. σ=0.15: 27.8 [22.4, 33.2] / 4.2 / 28.5 [23.9, 33.1]. σ=0.30: 3.5 / 0.9 / 3.1. HIPÓTESIS CAÍDA: la mosca NO aguanta el ruido mejor que una tabla con casillas finas (−0.7 pp [−9.3, 7.9] a σ=0.15). Sí le gana a la tabla que redondea a enteros (+23.5), pero porque redondear resulta contraproducente (convierte k=2 ruidoso en k=1 y fabrica certezas falsas), no por mérito de la mosca. Con ruido la mosca además comete marcas falsas (356 a σ=0.15; la tabla-fina 83). Conclusión: el buscaminas exige igualdades exactas (N=k, N=m); un 15% de ruido en el olfato hunde a cualquiera de 71% a 28%.


## Frente 3 cerrado · ¿importa el conectoma real? · 21 sept 2026 · runs/fly-wiring-001 y runs/fly-apl-001 · NO, Y USAR MÁS ESTRUCTURA REAL EMPEORA

3a (research/PROTOCOLO_CABLEADO.md). Real vs barajados en el régimen donde la arquitectura sí importa (primeras miles de partidas). 10 semillas pareadas (misma semilla de sentidos), 300 layouts reservados 9×9 con cascada, no letal, olfateos crudos. % victorias a 250 / 500 / 1k / 2k / 4k / 8k partidas:
real 30.8 / 42.3 / 47.1 / 53.4 / 68.9 / 71.4 · barajado completo 42.4 / 41.4 / 53.8 / 56.9 / 70.5 / 71.2 · solo entrada barajada 45.5 / 46.7 / 44.5 / 65.6 / 69.5 / 71.9 · solo salida barajada 40.4 / 44.4 / 59.1 / 66.8 / 67.8 / 69.2 · azar total 33.2 / 42.0 / 42.5 / 65.5 / 67.2 / 68.0.
Área temprana (250–2000), real − X: barajado −5.2 [−14.8, 4.3]; entrada −7.2; salida −9.3 (p=0.10); azar −2.4. Meseta (8k): real − barajado +0.2 [−3.4, 3.8]; real − azar +3.5 [−1.2, 8.1]. Ninguna diferencia a favor del cableado real; las nominales van en contra.
3b primera pieza. Inhibición APL REAL (cada APL suma la actividad Kenyon por sus sinapsis KC→APL reales e inhibe a cada Kenyon por sus sinapsis APL→KC reales; una ganancia global calibrada a 20% de actividad media) en lugar del corte exacto top-k:
apl-real 36.5 / 44.8 / 49.2 / 54.1 / 53.6 / 60.1 · topk-real 32.9 / 46.5 / 50.2 / 54.6 / 70.2 / 72.6 · apl-barajado 38.0 / 47.9 / 57.6 / 56.3 / 68.4 / 67.6 · apl-azar 43.4 / 45.2 / 58.9 / 62.6 / 68.7 / 68.3.
Meseta: apl-real − topk-real = −12.5 pp [−21.4, −3.6], p=0.011; apl-real − apl-barajado −7.5 (p=0.07); temprano apl-real − apl-azar −6.4 [−11.9, −0.8], p=0.03. Es decir: meter la inhibición real EMPEORA, y con los pesos APL reales es peor que con esos mismos pesos barajados. Lo único bueno: casi elimina las marcas falsas (1 vs 63).
Conclusión del frente: para esta tarea lo que sirve es la ARQUITECTURA (expansión dispersa a miles de unidades + lectura plástica gobernada por dopamina); el conectoma concreto de la mosca no aporta nada medible y sus heterogeneidades reales (grados, pesos APL) más bien estorban. Es coherente con la idea de que el cuerpo pedunculado es, a propósito, casi aleatorio en su entrada. NO probado: realimentación real MBON→DAN ni conexiones MBON→MBON; con esta evidencia no espero que cambien el veredicto y cuestan días. Se conserva top-k como configuración de trabajo.


## Frente 4 cerrado · error por pisada · 21 sept 2026 · hipótesis refutada, diagnóstico útil

Piloto dev benchmarks/fly-precision-pilot (4 semillas, 8k partidas, 300 tableros 9×9 con cascada + 80 de 16×16): rejilla de dispersión de Kenyon × ancho de sintonía. Victorias 9×9 / 16×16 / error por pisada en 16×16:
dispersión 0.05: 46–58 / 12–21 / 1.3–1.7% · 0.10: 58–69 / 21–33 / 1.0–1.4% · 0.20 (actual): 70–71 / 39–44 / 0.83–0.93% · 0.40: 44–63 / 10–24 / 1.5–2.7% con miles de marcas falsas.
Mi hipótesis (un código más disperso y fino confunde menos triples vecinos) era FALSA: más disperso es peor; el óptimo es la configuración actual (0.20; la sintonía entre 0.2 y 0.5 da igual dentro del ruido). No hay nada que confirmar en formal.
Diagnóstico de muertes (mosca de runs/fly-viz, 400 partidas 9×9 + 100 de 16×16 dev, lógica de un solo punto como vara): de 180 muertes, 153 (85%) fueron ADIVINANZAS FORZADAS (no existía ninguna casilla deducible como segura), 105 de ellas en casillas sin olor; 27 (15%) con una casilla segura disponible; solo 6 sobre una mina deducible. En 14,106 pisadas, 172 veces (1.2%) había una segura deducible y eligió otra.
Lectura: la deducción de un solo punto la mosca ya la hace casi perfecta. Lo que la separa de la política a mano y de un solver es CÓMO ADIVINA y las deducciones de dos pistas (frente 6), no la precisión de su código.


## Frente 5 cerrado · la compuerta de marcar, APRENDIDA · 21 sept 2026 · runs/fly-gate-001 · FUNCIONA: iguala al umbral puesto a mano

Protocolo research/PROTOCOLO_COMPUERTA_MARCA.md. Diagnóstico previo en pilotos dev (benchmarks/fly-gate-pilot, fly-gate-pilot2): con compuerta blanda (τ=1) θ converge a ≈ −6.3 DESDE AMBOS LADOS (desde −8 y desde −1): era un equilibrio, no un estancamiento — la compuerta blanda marca al azar casillas seguras cerca del umbral mientras explora, eso se castiga, y el umbral se retira. Con compuerta nítida (τ=0.25) sube a la zona del 3%. Partiendo de −1 (marcando de más) es inestable y en algunas semillas marca todo; hay que partir conservador.
Formal: olfateos crudos, no letal, densidad variable, 20k partidas, 10 semillas; evaluación letal con inicio en cascada, 500 layouts 9×9/12 + 250 de 7×7/7 reservados nuevos.
9×9: compuerta aprendida 73.5% IC95 [72.3, 74.8] (DE 1.8) · reflejo 3% puesto a mano 71.3% [68.0, 74.7] (DE 4.7) · dopamina simulada (sham) 44.3% [29.3, 59.2] · compuerta congelada en −12 26.2%.
   aprendida − reflejo +2.2 pp [−0.7, +5.2], p=0.12 (dentro del margen de equivalencia de ±5 prefijado; nominalmente mejor y con menos varianza) · aprendida − sham +29.3 [15.0, 43.6], p=0.001 · aprendida − congelada +47.3.
7×7: aprendida 84.0% [83.1, 84.9] · reflejo 83.9% · sham 66.0% · congelada 49.1%. Marcas falsas: aprendida 26 (9×9) y 4 (7×7), igual que el reflejo (27 y 4); sham 8,120.
θ final en las 10 semillas: entre −2.54 y −3.53, media −3.07. El valor puesto a mano equivalía a −3.48. La mosca DESCUBRIÓ sola, a partir de calor y azúcar retrasados, prácticamente el mismo umbral que yo había elegido a ojo. Con dopamina simulada θ no converge (de −10.5 a +3.9).
Aviso sobre el control: el brazo que llamé "never" NO es "nunca marca": LearnedMarker recorta θ a [−12, 4], así que quedó congelado en −12 y en 8 de 10 semillas todavía marca algo (5.1 marcas por partida de media). Es el mismo descuido del punto 4; el control limpio de esta corrida es el sham.
Qué sigue siendo puesto a mano en la compuerta: τ=0.25, el punto de partida conservador (−8), la relación calor −3 / azúcar +1 y la traza 0.9. El umbral ya no.
DECISIÓN: la compuerta aprendida pasa a ser la configuración de trabajo. Mejor mosca individual legítima hasta hoy: 73.5% en 9×9/12 con inicio justo.
Tamaño del premio del frente 6 (medido en dev, inicio en cascada): solver completo de restricciones 89.8% en 9×9 y 86.7% en 16×16; política a mano de un solo punto ≈ 79% y 62%; mosca ≈ 73% y 40%. Quedan ~16 pp en 9×9 y ~45 pp en 16×16 que solo se alcanzan cruzando pistas y adivinando con probabilidades.


## Frente 6 · pilotos sin mosca: ¿sirve recordar PARES de pistas? · 21 sept 2026 · SÍ, Y SOBREVIVE AL APRENDIZAJE POR RESULTADOS

Camino (a) elegido por Felipe: memoria de trabajo corta (el olfateo anterior + el paso dado), sin que la arena resuelva nada. Antes de construir la mosca, dos pilotos con un jugador que hace perfecta la lógica de un solo punto
(marca minas seguras, abre casillas seguras) y SOLO cuando está obligado a adivinar le pregunta a un adivinador que ve cantidades crudas:
  single = bolsa de triples (N,k,m) de las vecinas abiertas (lo que hoy puede representar la mosca) · pairs = single + por cada vecina abierta B, cada otra pista A a dos casillas o menos de B: (triple_B, triple_A, dónde está A respecto a B, dónde está B respecto a la casilla), canónico bajo las 8 simetrías.
Piloto 1, supervisado (experiments/pilot_pair_memory.py, benchmarks/pair-memory-pilot): 1.94 M de casillas etiquetadas de adivinanzas forzadas. Victorias 9×9 / 16×16 (inicio en cascada, 2000 y 320 partidas dev):
  single 75.7 / 62.2 · heurística a mano 81.7 / 68.8 · pairs 88.3 / 80.0 · solver exacto 89.8 / 83.8. Adivinanzas que salen seguras en 9×9: 87.4 / 90.9 / 93.7%. Pérdida logarítmica 0.492 → 0.375. 33,653 claves de pares en uso.
Piloto 2, SOLO RESULTADOS (experiments/pilot_pair_online.py, benchmarks/pair-online-pilot): tabla con regla delta que aprende únicamente del calor/azúcar de la casilla que pisa, entrenamiento no letal, 4 semillas. Victorias 9×9 (16×16):
  5k partidas: single 75.2 (53.5) · pairs 78.7 (60.1) — 20k: 75.5 (53.6) · 81.9 (62.2) — 80k: 77.6 (54.5) · 85.2 (71.6), y pairs sigue subiendo (52,715 claves).
Lectura: la información de pares recupera casi todo el hueco hasta el solver (88.3 vs 89.8 con supervisión) y la ventaja se conserva aprendiendo solo de resultados (+7.6 pp en 9×9 y +17 pp en 16×16 a 80k partidas), aunque necesita mucha más experiencia que los olfateos sueltos. Es el primer frente donde el techo sube de verdad.
Límites: (1) aquí la lógica de un solo punto está escrita a mano y el adivinador es una tabla; en la mosca todo se aprende a la vez y las claves pasan por 88 glomérulos y 4,064 Kenyon; (2) el espacio es de decenas de miles de combinaciones: es exactamente el régimen donde la generalización del circuito (×4 en eficiencia muestral, frente 2) podría pagar. Siguiente paso: construir la mosca con memoria de pares.


## Frente 6 · la mosca con memoria de pares · piloto dev · 21 sept 2026 · GANANCIA PEQUEÑA EN LA MOSCA, GRANDE EN UNA TABLA

Implementado: experiments/fly_senses.sniffs_pairs (PairStimulus: olfateos sueltos + por cada vecina abierta B, cada otra pista A a ≤2 casillas: N,k,m de B, N,k,m de A y la clase de movimiento tile→B, B→A canónica bajo 8 simetrías; 28 clases; todo crudo, m = marcas de la propia mosca), vía de 7 canales en MushroomBody.glomerular (6 concentraciones con sintonía en banda + firma aleatoria por clase de movimiento, repartidos en los mismos 88 glomérulos), PairFlyAgent (valor de casilla = suma de olfateos sueltos + media de pares; patrones Kenyon de pares guardados dispersos, memoria acotada a 60k) y PairTableAgent (sin cerebro). Prueba nueva: los pares son crudos y la bolsa de pares es invariante a rotar el tablero. Pruebas 14/14.
Piloto dev experiments/pilot_fly_pairs.py (todo se aprende a la vez, solo calor/azúcar, no letal, densidad variable, reflejo 3%, tasa que decae 1/(1+g/10000); 300 tableros 9×9 + 50 de 16×16 dev con cascada; 3–4 semillas). Victorias 9×9 (16×16):
  10k partidas: mosca sin pares 72.0 (46.7) · mosca con pares 72.2 (43.0) · tabla sin pares 67.6 (45.0) · tabla con pares 76.9 (55.5)
  20k: mosca 73.2 (43.3) · mosca+pares 75.4 (51.0) · tabla+pares 79.7 (61.0), con semillas en 83.7 y 82.0
  40k: mosca 71.3 (52.7) · mosca+pares 74.5 (58.5) · tabla 72.0 (50.0) · tabla+pares 75.7 (58.0) pero inestable: 76.0 / 63.7 / 81.7 / 81.3
Lectura honesta: la memoria de pares SÍ sube el techo cuando el aprendiz puede formar la conjunción exacta (tabla: hasta 82–84% en sus mejores semillas, contra ~72% sin pares; techo del solver 89.8%), pero en la mosca solo da ≈ +3 pp en 9×9 (74.5 vs 71.3) y ≈ +6 en 16×16, con pocas semillas y tableros: todavía sin confirmar. Aquí se invierte lo del frente 2: con pares, generalizar estorba. Hipótesis sin probar: el cuello son los 88 glomérulos — siete atributos por par repartidos en ~12 glomérulos cada uno, y cada Kenyon muestrea ~6 entradas, así que casi ninguna Kenyon liga los siete atributos y pares distintos se confunden.
Costo: la mosca con pares corre a ~0.2 s/partida (20× más lenta que sin pares) por los ~50k patrones de pares.
Pendiente de decidir: (i) darle a los pares una capa de entrada propia y más ancha (ya no sería el conectoma, pero el frente 3 mostró que el conectoma concreto no aporta); (ii) estabilizar el aprendizaje de pares (la tabla también oscila: una semilla cae de 77.7 a 63.7); (iii) dejar el frente aquí. Nada corriendo.


## ¿Sumar o promediar los olfateos? · 21 sept 2026 · runs/fly-aggregate-001 · PROMEDIAR SUBE ~9 pp EN 9 DE 10 MOSCAS, PERO 1 DE 10 SE HUNDE

Protocolo research/PROTOCOLO_PROMEDIAR_OLFATEOS.md. Motivo: el 85% de las muertes son adivinanzas forzadas y una heurística que promedia adivina mejor que un aprendiz que suma con la misma información. Cambio: valor de casilla = MEDIA de las respuestas a sus olfateos (normalización divisiva; ninguna información nueva); se aprende también con el patrón Kenyon medio.
Pilotos dev (benchmarks/fly-aggregate-pilot): media + reflejo fijo 3% = 33.7% con miles de marcas falsas (al promediar cambia la escala de cada olfateo y el umbral puesto a mano deja de servir; solo la compuerta aprendida se adapta: θ ≈ −6 en vez de −3). Media + compuerta aprendida: 76–81% en 5 de 6 semillas y una desbocada (θ = +3.9). Tope θ ≤ 0 añadido a la compuerta: 8 de 8 sanas en dev.
FORMAL (10 semillas, 20k partidas, layouts reservados nuevos con cascada: 500 de 9×9, 250 de 7×7, 150 de 16×16; política a mano 80.8 / 89.2 / 70.0):
  9×9: suma (oficial) 70.9% [68.8, 73.1] · suma con tope 72.4% [70.5, 74.3] · media con tope 73.3% [58.9, 87.7], por semilla 77.8 81.8 79.8 79.0 80.4 **16.2** 80.2 77.8 81.8 78.4. media − suma +2.4 pp [−11.4, +16.2], p=0.71: NO significativo por una semilla hundida.
  16×16: 45.0 / 48.5 / 53.2 (por semilla 54–64 y un 0.0). 7×7: 84.6 / 84.2 / 84.2 (86–91 y un 44.8).
  Sin la semilla 5 (dato descriptivo, NO el titular): media 79.7% en 9×9, 59.1% en 16×16, 88.6% en 7×7 — es decir, a la altura de la política a mano en 9×9.
  La semilla hundida no se desbocó hacia arriba (θ = −4.73, dentro del rango de las sanas): marca de más con ese umbral (3,489 de las marcas falsas de todo el brazo). El tope θ ≤ 0 no previene este modo de falla. El tope por sí solo (suma con tope vs suma) no cambia nada medible (+1.5, p=0.26) y elimina las marcas falsas.
Lectura: promediar es una mejora real y grande del modo de adivinar (9 de 10 moscas pasan de ~71% a ~80%), pero con ella vuelve una falla de 1 en 10 en la compuerta aprendida. Como titular honesto NO se puede adoptar todavía: el número oficial de una mosca sigue siendo 73.5% [72.3, 74.8] (runs/fly-gate-001, suma + compuerta aprendida). Pendiente: entender por qué esa semilla marca de más (hipótesis: con la media, los valores por olfateo de esa mosca quedan mal escalados respecto a su umbral) y hacer robusta la compuerta; o seleccionar moscas por su propio desempeño en entrenamiento (legítimo si no toca layouts reservados).


## Reporte consolidado · 21 sept 2026

research/REPORTE_FINAL_MOSCA.md + research/reporte-final-figura.png (barras de runs/fly-gate-001 y curva de eficiencia de runs/fly-vs-table-001). REPORTE_MOSCA_MARCADORA.md queda marcado como sustituido. Frente 6 (memoria de pares) detenido por decisión de Felipe: ampliar la entrada ya no sería la mosca. Frente 7 (cuerpo visible) sigue estacionado. Nada corriendo.


## Cada olfateo aprende por su cuenta y la mosca promedia · 21 sept 2026 · runs/fly-aggregate-002 · ADOPTADO: 81.3% EN 9×9

Diagnóstico de la semilla hundida de fly-aggregate-001 (reentrenada, scratchpad diag_mean.py): con la media, el error de aprendizaje es el del promedio de la casilla, así que un olfateo de mina segura se vuelve extremo para pesar (log-odds de seguridad −31 en N=1,k=1,m=0; en una mosca sana −24). Por el solape de Kenyon eso se contagia a olfateos parecidos casi siempre seguros: (2,5,1) quedó en −7.0, (2,4,1) en −9.1, (2,3,1) en −11.7 y dispararon marcas falsas (627 de 862, 542 de 813, 254 de 520). En la mosca sana (2,3,1) vale −2.4. La falla aparece tarde (a 12k partidas esa semilla estaba sana).
Arreglo: aggregate='meanown' — cada olfateo de la casilla pisada se condiciona por separado con el resultado (su propio error de predicción) y la decisión es la media de las opiniones. Sin información nueva. Variantes descartadas en piloto: suma/√n 74.2%; media + alarma del peor olfateo 67.9% (hunde una semilla). Piloto dev de 20k partidas con semillas 0–9 (benchmarks/fly-aggregate-pilot5): cada-olfateo 79.5% (mín 68.5, 42 marcas falsas); media normal 73.3% con la semilla 5 otra vez en 12%; suma 73.2%.
FORMAL (research/PROTOCOLO_CADA_OLFATEO_POR_SU_CUENTA.md): semillas 10–19, NO usadas al diseñar el arreglo; 20k partidas; layouts reservados nuevos con cascada (500 de 9×9, 250 de 7×7, 150 de 16×16); política a mano en esos mismos layouts 84.8 / 91.2 / 61.3 (este conjunto salió más fácil que otros: comparar solo dentro de la corrida).
  9×9: suma 76.3% [74.7, 77.8] · media 81.3% [80.0, 82.7] · cada-olfateo 81.3% [78.8, 83.8]. cada-olfateo − suma +5.1 pp [1.5, 8.7], p=0.011; media − suma +5.1 [3.5, 6.7], p=0.0001; cada-olfateo − media 0.0.
  7×7: 84.3 / 87.4 / 88.3 (+4.0 [0.7, 7.4], p=0.024). 16×16: 44.3 / 51.8 / 57.3 (+13.0 [3.6, 22.4], p=0.012; vs media +5.5, p=0.11).
  Marcas falsas: suma 70, media 145, cada-olfateo 0 en los tres tamaños. Ninguna semilla hundida en esta corrida en ningún brazo (la media normal se hundió en 2 de 36 moscas entrenadas a 20k entre todas las corridas; cada-olfateo en 0 de 20).
  θ aprendido con cada-olfateo: −3.0 a −4.2 (vuelve a la escala de la suma; con media normal −5 a −7.5).
DECISIÓN: configuración de trabajo = olfateos crudos + no letal + densidad variable + compuerta aprendida con tope θ ≤ 0 + cada olfateo por su cuenta con decisión promediada. Número de una mosca: 81.3% [78.8, 83.8] en 9×9/12 con inicio justo, a 3.5 pp de la política a mano en los mismos tableros (84.8%); en 16×16, 57.3% vs 61.3%. Pendiente: la visualización (scene/export_arena.py) sigue usando la mosca anterior (suma + reflejo 3%).
