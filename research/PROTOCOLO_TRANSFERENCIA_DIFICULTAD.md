# Evaluación por dificultad de las políticas preservadas

Se fija antes de evaluar cada `best-SEED-local.pt` de `runs/wide-reader-001`, semillas 20261002/3/4. Sin entrenamiento ni selección adicional. El fracaso de la lectura auxiliar de riesgo no justifica reemplazar estas políticas.

Presupuesto: 200 layouts nuevos por tamaño: 7×7/7 minas, 9×9/12, 12×12/22, 16×16/38; densidades aproximadamente 15%, no una comparación causal pura de tamaño. Los mismos 800 layouts se comparten entre las tres políticas: 2400 episodios, no 2400 tableros independientes. Exclusión de todos los layouts registrados previamente; apertura central segura idéntica y victorias automáticas fuera del denominador.

Principal: victorias autónomas por tamaño, con cada semilla visible e intervalo bootstrap por layout sobre la media de las tres políticas. Secundarias: elecciones seguras/oportunidades certificadas por el solver, elecciones que ignoran una segura, minas deducibles seleccionadas, derrotas con segura disponible y derrotas tras elegir un riesgo mínimo exacto de 50%. Estas últimas se informan separadas, sin transformarlas en victorias. El solver tiene límites y no certifica todas las deducciones posibles.

El maestro mide después de elegir; no decide ni filtra las acciones. Cerebro y encoder congelados por política; caché reiniciada entre minibloques para acotar RAM y no mezclar encoders. Batch 16, cuatro hilos para el operador disperso, un hilo Torch. Se guarda avance cada 50 partidas.

La evaluación servirá para orientar la distribución del siguiente entrenamiento hacia fallos observados. No se cambiará el agente servido ni se afirmará ventaja anatómica: son tres lectores/interfaces sobre un solo cerebro preentrenado.
