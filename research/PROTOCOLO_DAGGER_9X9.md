# DAgger con experiencia de 9×9

La evaluación corregida muestra 24.3% de victorias en 7×7 frente a 0.83% en 9×9, y una caída del aprovechamiento de seguras de 88.1% a 65.4%. Los datos históricos eran únicamente 5×5 y 7×7. Hipótesis: experiencia del propio agente en 9×9, etiquetada con deducciones públicas, mejora la transferencia sin modificar el cerebro ni la interfaz.

Piloto 20261002 desde `runs/wide-reader-001/best-20261002-local.pt`. Preservar pesos/Adam/RNG iniciales en ambos brazos. 3 rondas de 300 partidas autónomas 9×9/12 minas y 750 actualizaciones por ronda: 2250 en cada brazo. Control: 64 posiciones originales por batch. DAgger: 32 originales balanceadas por tamaño/categoría y 32 de experiencia nueva acumulada. Solo posiciones con seguras certificadas; el maestro etiqueta pero nunca actúa. Encoder/cerebro congelados, mapeo extendido verificado de difficulty-transfer-002. Son distribuciones de datos distintas, no minibatches idénticos.

Se fija de antemano el último paso2250 para ambos brazos; la pérdida del holdout pequeño se registra y sus mejores checkpoints se conservan como diagnóstico, pero no eligen los candidatos de esta prueba. Esta regla evalúa adaptación a un dominio nuevo sin seleccionarla exclusivamente por el dominio antiguo. No elegir después el checkpoint que gane más en test.

Principal: diferencia pareada DAgger−control en 500 partidas nuevas9×9. Secundarias: contra baseline preservado, retención en250 nuevas7×7, elecciones seguras/oportunidades, minas deducibles y derrotas con segura disponible. Intervalos bootstrap pareados por layout, una semilla/un cerebro. Colección900 y test750 disjuntos de todo layout registrado antes, incluida evaluación corregida. No mezclar porcentajes de tamaños en un único winrate.

Se generaliza el recolector existente para usar tamaño/minas de cada partida en entorno, contexto, solver y conversión de acciones. Verificar igualdad del recolector anterior en7×7 y consistencia de etiquetas/acciones públicas en9×9 antes de arrancar. Se conserva el agente servido. No se entrena encoder ni grafo en este piloto.
