# Pilotos de los tres métodos

64 partidas por método, 48 evaluaciones nuevas por método, sin maestro ni exploración. Mismo checkpoint inicial y mismos tableros de entrenamiento/evaluación. Implementación inicial de un hilo disperso: los tiempos no corresponden a la posterior optimización de cuatro hilos.

| Método | Clics | Updates | Segundos entrenamiento | Pico GB |
|---|---:|---:|---:|---:|
| dagger | 307 | 69 | 100.14 | 0.82 |
| qrdqn | 248 | 55 | 129.97 | 0.84 |
| hybrid | 381 | 88 | 308.9 | 0.81 |

## Victorias de evaluación

| Estrato | DAgger | QR-DQN | Combinado |
|---|---:|---:|---:|
| 5x5-3 | 1/8 | 1/8 | 0/8 |
| 7x7-7 | 0/8 | 0/8 | 0/8 |
| 9x9-10 | 0/8 | 0/8 | 0/8 |
| 12x12-24 | 0/8 | 0/8 | 0/8 |
| 16x16-40 | 0/8 | 0/8 | 0/8 |
| 16x16-56 | 0/8 | 0/8 | 0/8 |

Aperturas que ganaron automáticamente: [0, 0, 0].

Una semilla, ocho pruebas por estrato y solo 64 partidas de entrenamiento no establecen superioridad. Las victorias de entrenamiento con intervención del maestro no miden autonomía. La ayuda llegó a cero antes de terminar cada piloto. Los tres modificaron ganancias internas finitas; checkpoints y evaluación completos.

La reanudación actual 4→1 hilos fue exactamente equivalente a ejecución continua: benchmarks/resume_equivalence.json. 26 tests pasan. La corrida curriculum-001 siguió activa y no fue reconfigurada.
