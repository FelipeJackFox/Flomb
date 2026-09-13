# Ampliación del entrenamiento del lector espacial

Corrida spatial-extended-001, autorizada. Tres semillas20261002/3/4, cerebro y encoder congelados, mismos datos y arquitectura. Extender de1,000 hasta5,000 updates totales con batch64/Adam0.001 y selección solo por validación cada250 updates. Detener tras seis comprobaciones sin mejora acumulada de al menos0.0001; conservar el mejor checkpoint, incluyendo el de1,000 como candidato.

Los checkpoints previos guardaban el lector pero no Adam. Se reconstruyen exactamente las primeras1,000 actualizaciones con las semillas y minibatches originales, exigiendo igualdad bit a bit de todos los parámetros antes de continuar. Se guardan ahora estados completos de Adam y generador, además de último/mejor checkpoint.

Reservar antes del entrenamiento500 nuevos layouts7×7/7 minas excluyendo datasets y benchmarks anteriores. Comparación emparejada, cada semilla con su versión de1,000 updates en los mismos tableros. No consultar benchmarks pasados ni seleccionar por los nuevos resultados finales. 5×5 no se presenta como evaluación inédita.

Reutilizar caché de actividad para entrenamiento y memoización verificada durante evaluación. Mantener todos los checkpoints anteriores y no sustituir automáticamente el agente servido. La prueba estudia presupuesto del lector sobre un solo cerebro congelado, no nuevos cerebros ni ventaja anatómica.
