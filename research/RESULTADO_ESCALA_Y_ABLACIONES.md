# Escala de demostraciones y comparación de componentes

10,000 posiciones de entrenamiento de 1555 tableros; 1,000 posiciones reservadas de 146 tableros; 500 partidas finales independientes.

Las 500 partidas finales no aparecen en los conjuntos de los dos pilotos de arquitectura anteriores. Los splits nuevos no comparten layouts y los estados de entrenamiento/validación son distintos. Varias posiciones pertenecen al mismo tablero: los aciertos por posición son descriptivos, no 1,000 ensayos independientes.

## Control CNN con mayor presupuesto

3000 actualizaciones, batch 64, semillas [20260924, 20260925].

| Modelo / semilla | Posiciones antes → después | 5×5 victorias* | IC 95% | 7×7 victorias* | IC 95% | Entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| cnn / 20260924 | 351 → 972/1000 | 178/244 (73.0%) | 67.1%–78.1% | 108/250 (43.2%) | 37.2%–49.4% | 70.9 s |
| cnn / 20260925 | 301 → 970/1000 | 173/244 (70.9%) | 64.9%–76.2% | 111/250 (44.4%) | 38.4%–50.6% | 70.9 s |

*Se excluyen aperturas que ganaron automáticamente; IC de Wilson sobre partidas. Las semillas reutilizan las mismas partidas y no se suman como tableros nuevos.

| Modelo / semilla | Jugadas seguras elegidas / oportunidades | Muertes con jugada segura disponible | Muertes tras elegir riesgo mínimo exacto 50% |
|---|---:|---:|---:|
| cnn / 20260924 | 2554/2683 | 63 | 22 |
| cnn / 20260925 | 2714/2888 | 88 | 20 |

## Comparación emparejada de componentes

3000 actualizaciones, batch 16, semillas [20260924].

| Modelo / semilla | Posiciones antes → después | 5×5 victorias* | IC 95% | 7×7 victorias* | IC 95% | Entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| cnn / 20260924 | 351 → 977/1000 | 160/244 (65.6%) | 59.4%–71.3% | 111/250 (44.4%) | 38.4%–50.6% | 17.8 s |
| current / 20260924 | 367 → 767/1000 | 101/244 (41.4%) | 35.4%–47.7% | 4/250 (1.6%) | 0.6%–4.0% | 750.2 s |
| encoder_only / 20260924 | 392 → 855/1000 | 112/244 (45.9%) | 39.8%–52.2% | 8/250 (3.2%) | 1.6%–6.2% | 859.8 s |
| internal_only / 20260924 | 474 → 806/1000 | 91/244 (37.3%) | 31.5%–43.5% | 8/250 (3.2%) | 1.6%–6.2% | 1129.6 s |
| both / 20260924 | 496 → 828/1000 | 103/244 (42.2%) | 36.2%–48.5% | 15/250 (6.0%) | 3.7%–9.7% | 1274.3 s |

*Se excluyen aperturas que ganaron automáticamente; IC de Wilson sobre partidas. Las semillas reutilizan las mismas partidas y no se suman como tableros nuevos.

| Modelo / semilla | Jugadas seguras elegidas / oportunidades | Muertes con jugada segura disponible | Muertes tras elegir riesgo mínimo exacto 50% |
|---|---:|---:|---:|
| cnn / 20260924 | 2506/2635 | 67 | 18 |
| current / 20260924 | 1102/1420 | 206 | 11 |
| encoder_only / 20260924 | 1413/1730 | 204 | 16 |
| internal_only / 20260924 | 1126/1499 | 239 | 11 |
| both / 20260924 | 1397/1743 | 215 | 9 |

## Interpretación y límites

- El control CNN largo usa más ejemplos procesados por actualización; no se compara su costo o precisión como si tuviera el mismo presupuesto que las ablaciones.
- Dentro de `ablations`, CNN y cuatro variantes del conectoma reciben las mismas secuencias de minibatches. Las cuatro variantes del conectoma tienen un canal y la misma cabeza inicial: actual, solo encoder, solo dinámica interna, ambos. Mantienen todas las conexiones.
- El cambio interno incluye sesgo, retención, reinyección y transformación compartida. Aísla ese conjunto del encoder, no cada mecanismo individual. La ganancia original continúa aprendiendo en las cuatro variantes.
- Los ejemplos de entrenamiento tienen al menos una acción segura certificada. Las apuestas siguen sin objetivos explícitos de riesgo en esta fase.
- El contador 50% exige probabilidades exactas del solver y que la elegida tenga riesgo inmediato mínimo de 50%. No prueba optimalidad futura ni convierte automáticamente una derrota en victoria.
- Las oportunidades seguras las certifica un solver acotado: puede omitir deducciones. No se interpreta ausencia de certificado como imposibilidad matemática de una acción segura.
- No se ha aislado todavía el efecto de aumentar datos frente a aumentar actualizaciones, ni el aporte de la topología biológica frente a un grafo de control.
- Híbrido original pausado; no se migran estos checkpoints a su arquitectura.
