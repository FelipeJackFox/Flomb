# Repetición de interfaz conjunta

Autorizada14 septiembre2026. Coordinador `runs/joint-consistency-001`; nuevos pares `runs/joint-interface-20261003` y `runs/joint-interface-20261004`; piloto20261002 reutilizado sin reentrenar.

Repetir protocolo750updates,batch64,micro16,Adam lector heredado lr.001,encoder nuevo lr.0001,clip5 separado,misma secuencia dentro de cada par,datos originales10000/1000,control sololector/conjunto encoder+lector,conectoma y ganancias fijos. Cada semilla parte de su checkpoint cerebral elegido en la comparación de entradas. Validación0/250/500/750 mínima pérdida,empate temprano,incluye inicial.

Reservar500layouts nuevos7×7/7minas compartidos. Entrenar ambos pares y sellar los seis checkpoints antes de cualquier evaluación final. Evaluar baseline/control/conjunto de las tres semillas; deduplicar únicamente pesos de encoder+lector exactamente idénticos. Memo de actividad separado por hash del encoder; restaurar encoder correspondiente antes de inferencia.

Comparación principal: media de diferencias conjunto-control en las DOS semillas nuevas, pareada por tablero. Resumen de tres semillas secundario porque la primera motivó la repetición. Intervalos bootstrap condicionales a estos modelos,500tableros compartidos,un cerebro; no tratar1500 resultados como independientes. Excluir aperturas ganadoras automáticas. No reemplazar agente servido ni ampliar entrenamientos automáticamente.
