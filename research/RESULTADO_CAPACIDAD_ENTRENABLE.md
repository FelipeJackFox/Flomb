# Comparación de capacidad entrenable

Prueba exploratoria de arquitectura, no selección final de algoritmo RL.

120 actualizaciones por modelo y semilla; minibatches idénticos de 16; 512 posiciones de entrenamiento, 126 reservadas y 48 tableros de partidas autónomas. Los tres conjuntos no comparten distribuciones de minas; entrenamiento y posiciones reservadas tampoco comparten estados visibles.

| Modelo | Semilla | Reservadas antes → después | Entrenamiento | Victorias 5×5* | Victorias 7×7* | Segundos de entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 20260922 | 38/126 → 98/126 | 434/512 | 10/24 | 0/24 | 0.74 |
| current | 20260922 | 41/126 → 63/126 | 373/512 | 7/24 | 0/24 | 31.86 |
| expressive | 20260922 | 41/126 → 74/126 | 466/512 | 7/24 | 0/24 | 145.44 |
| cnn | 20260923 | 38/126 → 99/126 | 420/512 | 8/24 | 1/24 | 0.70 |
| current | 20260923 | 46/126 → 60/126 | 351/512 | 5/24 | 0/24 | 30.76 |
| expressive | 20260923 | 47/126 → 64/126 | 468/512 | 4/24 | 0/24 | 143.16 |
| random | referencia | — | — | 0/24 | 0/24 | — |
| visible_teacher | referencia | — | — | 18/24 | 21/24 | — |

*Se excluyen las partidas ganadas automáticamente por la apertura. Tiempo de entrenamiento excluye generación de datos, carga del grafo y evaluaciones.

Una acción legal uniforme tendría un acierto esperado del 35.3% en las posiciones reservadas. El maestro es una referencia práctica, no un límite óptimo demostrado.

## Decisiones reservadas por dificultad y tipo de deducción

| Modelo / semilla | 5×5 elemental | 5×5 relacional/exacta | 7×7 elemental | 7×7 relacional/exacta |
|---|---:|---:|---:|---:|
| cnn / 20260922 | 37/42 | 16/20 | 45/62 | 0/2 |
| current / 20260922 | 23/42 | 14/20 | 25/62 | 1/2 |
| expressive / 20260922 | 29/42 | 16/20 | 29/62 | 0/2 |
| cnn / 20260923 | 37/42 | 16/20 | 46/62 | 0/2 |
| current / 20260923 | 20/42 | 12/20 | 27/62 | 1/2 |
| expressive / 20260923 | 24/42 | 15/20 | 25/62 | 0/2 |

Elemental significa que el estado admite al menos una deducción segura por cierre cero/completo, incluido el total público de minas. Relacional/exacta significa que el maestro certifica seguridad más allá de ese cierre. No es una lectura del razonamiento interno del modelo.

## Comparaciones emparejadas

- 20260922, expressive frente a current: solo candidato acierta 24; solo actual acierta 13; McNemar exacto bilateral p=0.09887.
- 20260922, cnn frente a current: solo candidato acierta 41; solo actual acierta 6; McNemar exacto bilateral p=1.772e-07.
- 20260923, expressive frente a current: solo candidato acierta 23; solo actual acierta 19; McNemar exacto bilateral p=0.644.
- 20260923, cnn frente a current: solo candidato acierta 44; solo actual acierta 5; McNemar exacto bilateral p=7.597e-09.

Valores p exploratorios sin ajuste por comparaciones múltiples. Las semillas repiten los mismos tableros: no deben sumarse como nuevas observaciones independientes.

## Alcance y límites

- `current` conserva la propagación original y su clase de política lineal. Se validó equivalencia de features y gradientes; usa una cabeza categórica y Adam comunes a la prueba, no la optimización QR-DQN original.
- `expressive` conserva todo el grafo; aprende un codificador convolucional, dos canales por neurona, ganancias, sesgos, mezcla de canales y retención. Cambia varios componentes juntos. No aísla cuál ayuda.
- `cnn` es un control convencional pequeño. No está igualado en parámetros ni costo con el conectoma. Todos reciben los mismos datos, etiquetas, actualizaciones y tasa de aprendizaje; no es una búsqueda de hiperparámetros.
- El entrenamiento balancea tamaño y categoría. La evaluación conserva su distribución; solo hay dos posiciones relacionales/exactas 7×7. Ese estrato requiere ampliación antes de concluir.
- Los ejemplos supervisados son estados con al menos una jugada segura certificada. Las partidas autónomas sí pueden necesitar apuestas; esta prueba no entrena ni valida probabilidades de riesgo.
- El pico de memoria guardado es acumulado del proceso macOS, no consumo aislado de cada modelo.
- Los checkpoints incluyen parámetros y optimizador. La restauración del optimizador pasó una prueba de igualdad exacta; estos archivos no son compatibles directamente con el entrenador híbrido.
- Falta control con conectividad reconfigurada antes de atribuir una ventaja a la anatomía biológica.

Checkpoint híbrido preservado: True. Corrida principal sin reanudar.

Datos, resultados, manifiesto, fuentes exactas y checkpoints: `runs/capacity-diagnostic-001`.
