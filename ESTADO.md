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
está en `DISENO.md`; la visualización 3D no está construida. La actividad del modelo
actual son tasas artificiales, no spikes ni una reproducción fisiológica completa.
