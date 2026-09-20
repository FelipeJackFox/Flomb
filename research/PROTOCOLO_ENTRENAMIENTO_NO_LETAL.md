# Entrenamiento no letal sobre olfateos crudos (prefijado 19 sept 2026, antes de evaluar)

Hipótesis: el cuello de la mosca con sentidos crudos es la escasez de muestras de los olores que aprendió a temer (deja de pisarlos y la partida letal termina en la primera quemadura).
Intervención (solo en entrenamiento): pisar una mina da calor y enseña igual, pero la mosca sobrevive, deja su marca de estrés en esa casilla y sigue jugando. La EVALUACIÓN es siempre letal.
No es información nueva sobre el tablero que no tendría un jugador real: la mina que explotó queda a la vista. Piloto dev (benchmarks/fly-training-pilot): letal ~23–25% vs no letal ~50–57% en 9×9 a 20k partidas;
currículum por densidad sin ganancia → descartado y NO entra a la evaluación formal.
Brazos, 10 semillas (0..9), 20,000 partidas, sentidos crudos, resto idéntico a fly-raw-001: raw-real-soft (principal), raw-real (letal, mismo presupuesto de partidas), table-soft, additive-soft.
750 layouts reservados nuevos; unidad = semilla; contrastes pareados por semilla de raw-real-soft contra cada otro brazo; naive, política a mano, ens de 10 (solo como dato).
Preguntas: (1) ¿cuánto sube la mosca cruda? (2) ¿la ventaja del cerebro sobre tabla/aditiva sobrevive cuando sobran muestras, o era solo eficiencia muestral?
Nota de justicia: las partidas no letales son más largas, así que a igual número de partidas el brazo no letal ve más pisadas; se reporta el número de pisadas de entrenamiento por brazo.
