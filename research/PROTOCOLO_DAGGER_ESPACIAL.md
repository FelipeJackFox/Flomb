# DAgger espacial: piloto con cerebro congelado

Semilla de lector 20261002 fijada antes de observar resultados. Parte del checkpoint ampliado elegido por validación, con Adam restaurado. Cerebro, encoder y mapa óptico congelados. Tres rondas: 300 partidas autónomas 7×7/7 minas nuevas por ronda y 750 actualizaciones por variante. Batch 64, Adam lr 0.001, clipping 5.

DAgger beta=0: el lector ejecuta todas las acciones; el maestro etiqueta todas las posiciones visitadas con alguna jugada certificada segura, incluso cuando el lector iba a equivocarse. No se corrige el clic ni se continúa artificialmente tras morir. Se omiten posiciones sin acción certificada segura. Las etiquetas usan solo observación pública. Se acumulan todas las rondas; minibatches con 32 posiciones antiguas equilibradas por tamaño/categoría y 32 posiciones nuevas uniformes. La nueva distribución incluye tanto aciertos como errores.

Control: misma arquitectura y checkpoint inicial, mismo número de actualizaciones, batch 64 exclusivamente de datos originales equilibrados. Baseline sin actualizaciones. Selección final independiente por variante mediante mínima pérdida en las 1,000 posiciones antiguas de validación, comprobada cada 250 actualizaciones, incluyendo checkpoint inicial. La colección usa el estado actual de DAgger; la selección final puede recuperar una ronda anterior.

500 layouts finales reservados antes de entrenamiento y 900 de colección, disjuntos entre sí y respecto a datasets/benchmarks anteriores. Apertura central segura automática idéntica. Evaluación autónoma emparejada de baseline/control/DAgger solo al final. Reportar victorias, acciones seguras aprovechadas, minas conocidas seleccionadas y muertes con alternativa segura, con denominadores. Bootstrap emparejado por tableros, una sola semilla; resultado piloto, sin inferir beneficio anatómico ni rendimiento en otros tamaños.

Caché de actividad neuronal congelada y procesamiento de partidas en lotes de 16; no hay pistas directas al lector ni acciones compartidas mediante caché. Pesos, Adam, RNG, datasets y fuentes conservados. Comprobar hashes de originales al finalizar. No reemplazar agente servido en esta fase.
