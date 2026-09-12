# Benchmark de referencia ejecutado

Tiempo total: 11.46 segundos. 192 semillas por estrato, 1,152 tableros distintos, dos políticas pareadas. Semillas a partir de 3,000,000,000. No evalúa al modelo de mosca.

| Tablero/minas | Auto | Aleatorio sin auto | Solver sin auto | IC 95% solver | Fracción segura revelada | Clics seguros probados | Adivinaciones |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5x5-3 | 5 | 10/187 (5.3%) | 149/187 (79.7%) | 73.3%–84.8% | 85.9% | 768 | 290 |
| 7x7-7 | 0 | 0/192 (0.0%) | 134/192 (69.8%) | 63.0%–75.8% | 74.0% | 1525 | 353 |
| 9x9-10 | 0 | 0/192 (0.0%) | 153/192 (79.7%) | 73.4%–84.8% | 81.1% | 2597 | 332 |
| 12x12-24 | 0 | 0/192 (0.0%) | 96/192 (50.0%) | 43.0%–57.0% | 60.2% | 5024 | 490 |
| 16x16-40 | 0 | 0/192 (0.0%) | 117/192 (60.9%) | 53.9%–67.6% | 68.3% | 9770 | 432 |
| 16x16-56 | 0 | 0/192 (0.0%) | 19/192 (9.9%) | 6.4%–14.9% | 33.6% | 5983 | 740 |

Los intervalos expresan incertidumbre sobre estas tasas, no prueban aprendizaje del modelo neuronal. El solver aplica reglas fijas. Los grandes con 56 minas siguen siendo difíciles: hay decisiones inciertas y el límite de enumeración también omite deducciones. Una probabilidad heurística no debe mostrarse como certeza.

Todos los detalles por episodio y supervivencia de adivinaciones están en `results/benchmark.json`. La repetición exacta y los límites están en `README.md`.
