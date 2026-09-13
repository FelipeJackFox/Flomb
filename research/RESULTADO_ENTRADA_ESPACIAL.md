# Resultado de la comparación espacial

Prueba exploratoria terminada. Ambas variantes entrenaron el cerebro completo200actualizaciones por semilla, con256posiciones y minibatches emparejados. Dos semillas. Codificador local3x3 fijo y sin parámetros adicionales; no es una CNN aprendida. Híbrido pausado en14,384: SHA256 confirmado sin cambios.

| Semilla | Entrada | Conocidas /256 | Reservadas /128 | Observaciones nuevas /126 | Victorias5x5 | Victorias7x7 |
|---|---|---:|---:|---:|---:|---:|
|20260920|local|206|67|66|10/62|0/64|
|20260920|raw|209|66|65|13/62|0/64|
|20260921|local|209|61|60|11/62|0/64|
|20260921|raw|223|59|58|12/62|1/64|

## Comparación emparejada

- Semilla20260920: solo local acierta3; solo actual acierta2; McNemar exacto bilateral p=1.0000.
- Semilla20260921: solo local acierta6; solo actual acierta4; McNemar exacto bilateral p=0.7539.

## Límites e interpretación

Los splits son disjuntos por distribución real de minas. Dos posiciones reservadas tienen el mismo estado visible que alguna de entrenamiento pese a pertenecer a tableros distintos; se muestra también el análisis excluyéndolas. Las partidas completas usan un tercer conjunto disjunto. Ninguna recompensa, etiqueta ni entrada del modelo consulta ubicaciones ocultas. El hash de minas se utiliza exclusivamente al separar los datasets.

No se deben sumar las dos semillas como256tableros independientes: se repiten los mismos128. Tampoco equivale acertar posiciones del solver a ganar trayectorias autónomas. Una ausencia de ventaja aquí solo se refiere a este promedio local fijo, presupuesto y datos; no descarta codificadores convolucionales aprendidos ni demuestra que la conectividad sea irrelevante. Ver PROTOCOLO_ENTRADA_ESPACIAL.md para detalles.

## Decisión

No adoptar este codificador local fijo: no hay ventaja clara frente a la entrada actual. Aciertos reservados mejoran solo1y2posiciones; partidas completas favorecen ligeramente la entrada actual en ambas semillas. Es evidencia exploratoria, no equivalencia demostrada. Ambas quedan lejos de resolver7x7. Mantener híbrido pausado y no lanzar entrenamiento largo con este cambio.
