# Repetición con mapeo de actividad explícito

La primera evaluación por tamaño detectó un defecto de implementación: `activity_map` solo agrupaba actividad para 5×5 y 7×7. La ruta usada por el lector no tenía la validación que sí tenía `RetinaPolicy.forward`, de modo que devolvía ceros para los tamaños mayores. Las 1800 partidas de 9/12/16 de `difficulty-transfer-001` quedan invalidadas como prueba de generalización; se preservan como evidencia del fallo. 7×7 sí era válido. No se entrenaron pesos con estos mapas vacíos.

Corrección: generar el mismo mapeo por cuatro vecinos más cercanos y ponderación de distancia desde las anotaciones originales, ahora también para 9/12/16. Registrar las tablas nuevas como buffers no persistentes después de cargar el checkpoint histórico; las tablas 5/7 deben coincidir exactamente. `activity_map` recorre los tamaños presentes y rechaza tamaños sin mapeo. No se cambian parámetros aprendidos ni checkpoint.

Prueba previa con grafo real: igualdad exacta de actividad con el código histórico en 5/7; actividad finita, no vacía y distinta ante dos observaciones distintas en 9/12/16; padding cero y rechazo temprano cuando falta mapeo. Fuente y verificación en `experiments/test_extended_activity_mapping.py` y `benchmarks/extended-mapping-verification.json`.

Repetición `runs/difficulty-transfer-002`: idénticas tres políticas, 200 layouts nuevos por tamaño, 800 compartidos/2400 episodios. Excluir también todos los layouts de la evaluación defectuosa. Misma apertura, métricas, máscaras y presupuesto que el protocolo anterior; sin entrenamiento ni intervención del maestro. El test mide transferencia de políticas entrenadas en 5/7 usando una interfaz ampliada, no entrenamiento previo en grandes. Los IC bootstrap [0,0] cuando no hay victorias son degenerados y no prueban probabilidad verdadera cero.

Comando: `.venv/bin/python -m experiments.evaluate_difficulty_transfer --out runs/difficulty-transfer-002 --expand-mapping`. Cierre: `.venv/bin/python -m experiments.report_difficulty_transfer --out runs/difficulty-transfer-002`. Conservar agente servido sin cambios.
