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

## Continuación iniciada

Se lanzó inmediatamente una continuación de 2,048 episodios desde `checkpoint-000128.npz`.
Objetivo total: 2,176 episodios. El proceso termina al completar esa cantidad y evaluar.
Estimación inicial aproximada: 15–20 minutos; puede variar con longitud de partidas.
Este documento es una instantánea de lanzamiento, no prueba de que el proceso siga vivo.

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

## Pendiente

Evaluar la continuación, ampliar evaluación con nuevas semillas, probar cerebro
congelado y modelos de control antes de afirmar ventaja del conectoma. El diseño visual
está en `DISENO.md`; la visualización 3D no está construida. La actividad del modelo
actual son tasas artificiales, no spikes ni una reproducción fisiológica completa.
