# Punto 3: sentidos crudos de densidad + densidad variable (prefijado 20 sept 2026, antes de evaluar)

Base: olfateos crudos (N,k,m), entrenamiento no letal (adoptado tras fly-raw-002), evaluación siempre letal, 20,000 partidas, unidad = semilla (10).
Sentido nuevo (experiments/fly_senses.sniffs_far): SOLO en una casilla sin vecinas abiertas (nada fuerte que lo enmascare) la mosca huele tres cantidades crudas de toda la arena:
M = el contador de minas tal como se muestra (nunca lo reducimos), C = cuánto olor "tapado" hay en total (resolución de Weber, pasos de 25%), P = cuánta feromona propia hay en total.
Sin divisiones ni restas. El campo lejano nunca dispara el reflejo de marca. Un contexto igual para todas las casillas no cambiaría ninguna decisión (las valencias se suman), por eso solo existe donde no hay olor cercano.
Entrenamiento variable: densidad uniforme 8–25% por partida, misma agenda para todos los brazos. Fijo: 15% (12 minas en 9×9, 7 en 7×7).
Brazos: mosca y tabla sin cerebro × {lejano+variable, cercano+variable, cercano+fijo}. "Lejano+fijo" se omite: a densidad fija M es constante y no informa.
Evaluación: layouts reservados nuevos en tres densidades — 10% (8/5 minas), 15% (12/7), 20% (16/10) —, 300 de 9×9 y 150 de 7×7 por densidad, sin aperturas que ganen solas; política a mano como techo.
Principal: mosca lejano+variable − mosca cercano+variable, 9×9, promedio de las tres densidades, pareado por semilla. Secundarios: variable − fijo (¿cuánto cuesta en 15% y cuánto ayuda fuera?),
lo mismo para la tabla, mosca − tabla, fracción de pisadas a ciegas, marcas falsas.
Piloto dev (1 semilla, 5k partidas, 100 tableros por densidad): solo sirvió para comprobar que el código corre; demasiado ruido para orientar.
