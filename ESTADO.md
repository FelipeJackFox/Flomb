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
